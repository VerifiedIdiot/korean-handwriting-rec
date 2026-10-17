"""손글씨 단어로 korean PP-OCRv5 mobile rec 의 CTC 경로를 미세조정한다 — Mac GPU(MPS), PyTorch.

PaddlePaddle 은 Mac GPU 를 쓰지 못해 CPU 학습이 반복 1회(128장)에 100초를 넘었다 (2026-10-16 실측).
그래서 사전학습 가중치를 PyTorch 로 옮겨(convert_check.py — ONNX 와 같은 답 확인) CTC 경로만 학습한다.
보조 학습 머리(NRTR)는 추론 ONNX 에 없고 결과에 쓰이지 않아 뺀다.

  .venv/bin/python -I train_rec.py --train data/train_list.txt --val data/val_list.txt --out output/hw1
  .venv/bin/python -I train_rec.py --train ... --val ... --out output/bench --bench 40    # 속도만 잰다
  .venv/bin/python -I train_rec.py --train ... --val ... --out output/hw1-resume --resume output/hw1/snap-7500.pt --start-step 7500 --warmup 200  # 이어 학습 (같은 --seed·--batch)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# -I 로 돌리면 스크립트 폴더가 모듈 경로에 없다 (convert_check 를 같은 폴더에서 읽는다)
sys.path.insert(0, str(Path(__file__).resolve().parent))
# MPS 는 입력 모양마다 수 GB 를 캐시에 쥔다. 기본 문턱(권장 작업 메모리의 1.4배)에서는 학습 프로세스가 40GB 를 넘겨
# 48GB 장비가 스왑으로 멈췄다 (2026-10-16). 이 비율을 넘으면 쥔 캐시를 놓게 한다 — torch 를 불러오기 전에 정해야 한다
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from convert_check import CHARS, decode, load_torch

INDEX = {char: index for index, char in enumerate(CHARS) if index}
# RapidOCR·PaddleOCR 인식 입력과 같다: 높이 48, 너비는 비율대로 (상한 320)
HEIGHT, MAX_WIDTH = 48, 320
# 16 이면 입력 모양이 20가지라 MPS 캐시가 42GB 까지 쌓여 스왑으로 학습이 멈췄다 (2026-10-16, 48GB 장비)
WIDTH_STEP = 32
# MPS 는 쓴 메모리를 크기별로 캐시에 쥐고 있다 — 이 반복마다 돌려준다
EMPTY_CACHE_EVERY = 1
# 인쇄체를 섞어 학습한다 — 같은 인식 모델이 서류의 인쇄 글자도 읽는다. 손글씨만 보면 인쇄 글자 인식이 무너질 수 있다.
# 고딕(굴림·돋움 계열)과 명조(바탕 계열). 명조·애플고딕은 KS X 1001 음절만 있어, 그 밖의 음절은 산돌고딕네오(전 음절)로 그린다
_FULL_FONTS = tuple(("/System/Library/Fonts/AppleSDGothicNeo.ttc", index) for index in (0, 2, 4, 6))
_KS_FONTS = (("/System/Library/Fonts/Supplemental/AppleGothic.ttf", 0), ("/System/Library/Fonts/Supplemental/AppleMyungjo.ttf", 0))


class CtcRec(nn.Module):
    """CTC 경로만. 학습 때는 logits, 평가·내보내기 때는 softmax — 지금 ONNX 와 같은 출력 [B, T, 11947]."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.backbone, self.encoder, self.head = net.backbone, net.head.ctc_encoder, net.head.ctc_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.backbone(x)))

    def train(self, mode: bool = True) -> "CtcRec":
        super().train(mode)
        # 앞단은 늘 추론 동작: 학습 모드의 adaptive_avg_pool2d([1, 40]) 는 너비 320 입력을 전제로 한다
        # (그때는 추론의 avg_pool2d([3, 2]) 와 같은 계산). 정규화 통계도 사전학습 값으로 둔다 —
        # 손글씨만 보고 통계가 바뀌면 인쇄 글자 인식이 흔들린다. 기울기는 그대로 흐른다.
        self.backbone.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


def build_model() -> CtcRec:
    """사전학습 가중치 + 앞단 합성곱 가지 합치기(rep).

    정규화 통계를 고정하므로 가지들은 합성곱 하나와 같은 계산이다 (합치기 전후 출력 차이 2e-6).
    가지를 그대로 두면 Mac GPU 학습이 4배 느렸다 (2026-10-16 실측: 너비 128 배치 128 기준 1.23초 → 0.30초).
    """
    model = CtcRec(load_torch())
    for module in model.backbone.modules():
        if module is not model.backbone and callable(getattr(module, "rep", None)):
            module.rep()
    return model


def resized_width(width: int, height: int) -> int:
    return min(MAX_WIDTH, max(8, math.ceil(HEIGHT * width / height)))


class Words(Dataset):
    """라벨 목록 한 줄 = 상대 경로 \\t 글 [\\t 높이 48 기준 너비]. 경로는 목록 파일 폴더 기준."""

    def __init__(self, list_path: Path, *, augment: bool) -> None:
        self.items = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or not parts[1]:
                continue
            width = int(parts[2]) if len(parts) > 2 else 0
            self.items.append((str(list_path.parent / parts[0]), parts[1], width))
        self.augment = augment
        # 인쇄체 표본 가운데 문장부호·숫자 형식(_decorate)으로 바꿔 그릴 비율
        self.printed_forms = 0.0

    def widths(self) -> list[int]:
        out = []
        for path, _, width in self.items:
            if not width:
                height, raw = cv2.imread(path).shape[:2]
                width = resized_width(raw, height)
            out.append(width)
        return out

    def __len__(self) -> int:
        return len(self.items)

    def printed_width(self, index: int) -> int:
        """인쇄체로 그렸을 때 높이 48 기준 너비의 어림값 — 한글은 글자 높이만큼, 그 밖의 글자는 그 절반쯤."""
        text = self.items[index][1]
        return min(MAX_WIDTH, 8 + sum(46 if "가" <= char <= "힣" else 26 for char in text))

    def __getitem__(self, key: int | tuple[int, bool]) -> tuple[np.ndarray, list[int], str]:
        # 표본기가 (번호, 인쇄체로 그릴지)를 정해 넘긴다 — 인쇄체는 손글씨 상자보다 넓어 따로 너비별로 묶어야 한다
        index, printed = key if isinstance(key, tuple) else (key, False)
        path, text, _ = self.items[index]
        if printed and np.random.rand() < self.printed_forms:
            text = _decorate(text)
        image = _render(text) if printed else cv2.imread(path)
        if self.augment:
            image = _augment(image)
        height, width = image.shape[:2]
        resized = cv2.resize(image, (resized_width(width, height), HEIGHT)).astype(np.float32) / 255.0
        return ((resized - 0.5) / 0.5).transpose(2, 0, 1), [INDEX[char] for char in text], text


def _decorate(text: str) -> str:
    """서류에 흔한 문장부호·숫자 형식. 손글씨 단어에는 거의 없어(쉼표 0.04%, 빗금 0.07%) 7,500회 모델이
    인쇄된 쪽 표기 "1/2"를 통째로 놓치고 "(인쇄)"의 괄호·마침표를 버렸다 (2026-10-16 스캔 서류)."""
    rng = np.random
    num = lambda low, high: str(rng.randint(low, high))  # noqa: E731
    forms = (
        lambda: f"({text})", lambda: f"{text}.", lambda: f"{text},", lambda: f"{text}:", lambda: f"[{text}]",
        lambda: f'"{text}"', lambda: f"{text}·{text}", lambda: f"{num(1, 10)}/{num(1, 10)}",
        lambda: f"총 {num(1, 10)}면 중 제{num(1, 10)}면", lambda: f"{num(1990, 2031)}/{num(1, 13).zfill(2)}/{num(1, 29).zfill(2)}",
        lambda: f"{num(1990, 2031)}. {num(1, 13)}. {num(1, 29)}.", lambda: f"{num(1990, 2031)}-{num(1, 13).zfill(2)}-{num(1, 29).zfill(2)}",
        lambda: f"{int(num(1, 100000)) * 1000:,}원", lambda: f"({num(1, 61)})개월", lambda: f"010-{num(1000, 10000)}-{num(1000, 10000)}",
        lambda: f"{num(1, 30)}~{num(30, 100)}%", lambda: f"제{num(1, 20)}조({text})",
    )
    return forms[rng.randint(len(forms))]()


def _render(text: str) -> np.ndarray:
    """글을 인쇄체로 그린 단어 이미지 (BGR). 배경·글자 밝기와 여백을 조금씩 바꾼다."""
    rng = np.random
    try:
        text.encode("euc-kr")
        fonts = _FULL_FONTS + _KS_FONTS
    except UnicodeEncodeError:
        fonts = _FULL_FONTS
    path, face = fonts[rng.randint(len(fonts))]
    font = ImageFont.truetype(path, size=int(rng.randint(28, 49)), index=face)
    left, top, right, bottom = font.getbbox(text)
    pad = int(rng.randint(2, 9))
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), color=int(rng.randint(200, 256)))
    ImageDraw.Draw(canvas).text((pad - left, pad - top), text, font=font, fill=int(rng.randint(0, 81)))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_GRAY2BGR)


def _augment(image: np.ndarray) -> np.ndarray:
    """사진·스캔 차이를 흉내 내는 약한 변형. 글자 모양을 바꾸지는 않는다."""
    rng = np.random
    if rng.rand() < 0.3:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rng.uniform(-2, 2), 1.0)
        matrix[0, 1] += rng.uniform(-0.1, 0.1)
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if rng.rand() < 0.5:
        image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
    if rng.rand() < 0.3:
        # 스캔·복사 해상도: 줄였다가 되돌린다. 작은 인쇄 글자의 빗금·쉼표가 뭉개진 모양 — 학습 모델이 스캔 서류의
        # 쪽 표기 "1/2"를 "112"로 읽었다 (2026-10-17, 깨끗하게 그린 "1/2"는 맞게 읽음)
        height, width = image.shape[:2]
        scale = rng.uniform(0.35, 0.8)
        small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if rng.rand() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if rng.rand() < 0.2:
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 80))])
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """배치 안 가장 넓은 너비로 0(정규화 뒤 회색)을 채운다 — RapidOCR 추론과 같은 채움.

    너비는 WIDTH_STEP 단위로 올린다. Mac GPU 는 입력 모양마다 계산 그래프를 새로 만들어, 너비가 제각각이면 느리다.
    """
    width = min(MAX_WIDTH, -(-max(x.shape[2] for x, _, _ in batch) // WIDTH_STEP) * WIDTH_STEP)
    images = np.zeros((len(batch), 3, HEIGHT, width), np.float32)
    for row, (x, _, _) in enumerate(batch):
        images[row, :, :, : x.shape[2]] = x
    targets = torch.tensor([index for _, label, _ in batch for index in label], dtype=torch.long)
    lengths = torch.tensor([len(label) for _, label, _ in batch], dtype=torch.long)
    return torch.from_numpy(images), targets, lengths, [text for _, _, text in batch]


class WidthBuckets(Sampler):
    """비슷한 너비끼리 배치로 묶는다. 짧은 단어를 긴 단어 너비로 채우면 계산이 몇 배가 된다.

    인쇄체로 그릴 표본도 여기서 정한다 — 인쇄체는 같은 글의 손글씨 상자보다 넓어서, 섞어 묶으면 배치 대부분이
    가장 넓은 인쇄체 너비로 채워졌다 (실측: 초당 140장, 가른 뒤보다 2배 이상 느림). (번호, 인쇄체 여부)를 낸다.
    """

    def __init__(self, dataset: "Words", printed: float, batch_size: int, seed: int) -> None:
        self.widths = dataset.widths()
        self.printed_widths = [dataset.printed_width(i) for i in range(len(dataset))] if printed else []
        self.printed, self.batch_size, self.seed, self.epoch = printed, batch_size, seed, 0
        # 이어 학습: 첫 회차에서 이미 학습한 배치 수 (같은 시드·배치 크기면 순서가 같다)
        self.skip = 0

    def __len__(self) -> int:
        return len(self.widths) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        keys = [(i, rng.random() < self.printed) for i in range(len(self.widths))]
        rng.shuffle(keys)
        width = lambda key: self.printed_widths[key[0]] if key[1] else self.widths[key[0]]  # noqa: E731
        chunk, batches = self.batch_size * 64, []
        for start in range(0, len(keys), chunk):
            part = sorted(keys[start : start + chunk], key=width)
            batches += [part[i : i + self.batch_size] for i in range(0, len(part) - self.batch_size + 1, self.batch_size)]
        rng.shuffle(batches)
        yield from batches[self.skip :]


def _seed_worker(_: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Words, device: torch.device, limit: int) -> dict[str, float]:
    model.eval()
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    indices = sorted(indices[:limit], key=lambda i: dataset.items[i][2] or 0)
    correct = edits = length = 0
    for start in range(0, len(indices), 256):
        images, _, _, texts = collate([dataset[i] for i in indices[start : start + 256]])
        probs = model(images.to(device)).float().cpu().numpy()
        for row, truth in enumerate(texts):
            text = decode(probs[row])
            correct += text == truth
            edits += Levenshtein.distance(text, truth)
            length += len(truth)
    return {"word_acc": correct / len(indices), "cer": edits / length, "words": len(indices)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-words", type=int, default=5000)
    parser.add_argument("--printed", type=float, default=0.2, help="인쇄체로 그려 섞는 비율")
    parser.add_argument("--printed-forms", type=float, default=0.0, help="인쇄체 표본 중 문장부호·숫자 형식으로 바꿔 그릴 비율")
    parser.add_argument("--bench", type=int, default=0, help="이 반복 수만 돌려 속도를 잰다")
    parser.add_argument("--compile", action="store_true", help="torch.compile 로 연산을 합친다 (Mac GPU 는 실험 기능)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="이어 학습할 가중치 (best.pt·stepN.pt)")
    parser.add_argument("--start-step", type=int, default=0, help="이어 학습: 이미 마친 반복 수 — 그만큼 배치를 건너뛰고 학습률도 거기서 잇는다")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps")
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_model()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
    model = model.to(device)
    # 컴파일본은 같은 매개변수를 쓴다 — 저장은 원래 모듈로 해 키 이름이 바뀌지 않게 한다
    net = torch.compile(model) if args.compile else model
    train_set, val_set = Words(args.train, augment=True), Words(args.val, augment=False)
    train_set.printed_forms = args.printed_forms
    sampler = WidthBuckets(train_set, args.printed, args.batch, args.seed)
    sampler.skip = args.start_step
    loader = DataLoader(
        train_set, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
        persistent_workers=args.workers > 0, worker_init_fn=_seed_worker, prefetch_factor=4 if args.workers else None,
    )
    total = args.bench or max(1, int(len(sampler) * args.epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # 이어 학습이면 학습률 곡선은 start_step 지점에서 잇고, 옵티마이저 상태가 없으니 워밍업은 다시 한다
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup)
        * 0.5 * (1 + math.cos(math.pi * min(args.start_step + step, total) / total)),
    )
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    log = (args.out / "log.jsonl").open("a", encoding="utf-8")

    def record(entry: dict) -> None:
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.flush()

    if not args.bench:
        best = evaluate(net, val_set, device, args.eval_words)
        record({"step": args.start_step, "val": best})
        torch.save(model.state_dict(), args.out / "best.pt")
    step, seen, started = args.start_step, 0, time.perf_counter()
    while step < total:
        for images, targets, lengths, _ in loader:
            net.train()
            log_probs = net(images.to(device)).log_softmax(-1).permute(1, 0, 2)
            steps = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
            loss = ctc(log_probs, targets.to(device), steps, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            step, seen = step + 1, seen + images.shape[0]
            if step % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()
            if step % 50 == 0 or step == total:
                elapsed = time.perf_counter() - started
                # 쓰는 양(텐서)과 쥔 양(캐시 포함)을 함께 남긴다 — 쓰는 양이 늘면 누수, 쥔 양만 늘면 캐시
                record({"step": step, "of": total, "loss": round(float(loss), 4), "lr": schedule.get_last_lr()[0],
                        "samples_per_s": round(seen / elapsed, 1), "elapsed_min": round(elapsed / 60, 1),
                        "mps_used_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
                        "mps_held_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2)})
            if not args.bench and (step % args.eval_every == 0 or step == total):
                result = evaluate(net, val_set, device, args.eval_words)
                torch.mps.empty_cache()
                record({"step": step, "val": result})
                torch.save(model.state_dict(), args.out / f"step{step}.pt")
                if result["word_acc"] > best["word_acc"]:
                    best = result
                    torch.save(model.state_dict(), args.out / "best.pt")
            if step >= total:
                break
        sampler.epoch += 1
        sampler.skip = 0
    torch.save(model.state_dict(), args.out / "last.pt")
    record({"done": True, "best": None if args.bench else best})


if __name__ == "__main__":
    main()
