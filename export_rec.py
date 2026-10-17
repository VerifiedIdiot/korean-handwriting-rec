"""학습한 CTC 경로를 ONNX 로 내보내고 확인한다. 입력 x [N, 3, 48, W] · 출력 [N, T, 11947] softmax 는
지금 쓰는 인식 모델(PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx)과 같다 — RapidOCR 등에 그대로 끼운다.

  .venv/bin/python -I export_rec.py --checkpoint output/hw1/best.pt --out output/hw1/rec_korean_hw.onnx --check data/val_list.txt
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from rapidfuzz.distance import Levenshtein

from convert_check import ONNX, decode, preprocess
from train_rec import build_model


# AI Hub 053 쪽 이름에 든 종류 표시 (IMG_OCR_53_4PO_09451 …)
_KINDS = {"4PO": "종이·서식", "4PR": "종이·자유", "4TO": "태블릿·서식", "4TR": "태블릿·자유"}


def read(session: ort.InferenceSession, rows: list[tuple[Path, str]]) -> list[tuple[bool, int, int]]:
    """한 단어씩 RapidOCR 와 같은 전처리로 읽는다 (배치 채움 없음). 단어마다 (맞음, 편집 거리, 정답 길이)."""
    results = []
    for path, truth in rows:
        text = decode(session.run(None, {"x": preprocess(cv2.imread(str(path)))})[0][0])
        results.append((text == truth, Levenshtein.distance(text, truth), len(truth)))
    return results


def accuracy(results: list[tuple[bool, int, int]]) -> tuple[float, float]:
    """단어 정확도와 CER."""
    return sum(r[0] for r in results) / len(results), sum(r[1] for r in results) / sum(r[2] for r in results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path, required=True, help="확인할 라벨 목록 (학습에 안 쓴 분할)")
    parser.add_argument("--words", type=int, default=3000)
    parser.add_argument("--train", type=Path, help="학습 라벨 목록 — 주면 학습에 없던 단어만 따로도 잰다")
    args = parser.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 48, 320),),
        str(args.out),
        input_names=["x"],
        output_names=["fetch_name_0"],
        dynamic_axes={"x": {0: "N", 3: "W"}, "fetch_name_0": {0: "N", 1: "T"}},
        opset_version=17,
        dynamo=False,
    )
    exported = ort.InferenceSession(str(args.out))
    lines = [line.split("\t") for line in args.check.read_text(encoding="utf-8").splitlines() if line]
    random.Random(1).shuffle(lines)
    rows = [(args.check.parent / parts[0], parts[1]) for parts in lines[: args.words]]
    diff = 0.0
    for path, _ in rows[:100]:
        x = preprocess(cv2.imread(str(path)))
        with torch.no_grad():
            diff = max(diff, float(np.abs(model(torch.from_numpy(x)).numpy() - exported.run(None, {"x": x})[0]).max()))
    before, after = read(ort.InferenceSession(str(ONNX)), rows), read(exported, rows)
    print(f"ONNX 내보내기: {args.out} ({args.out.stat().st_size / 1e6:.1f}MB, sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()})")
    print(f"PyTorch 대 내보낸 ONNX 확률 최대 차이 (100단어): {diff:.2e}")
    groups = {"손글씨 전체": list(range(len(rows)))}
    for code, kind in _KINDS.items():
        groups[f"  {kind}"] = [i for i, (path, _) in enumerate(rows) if f"_{code}_" in path.name]
    groups["  학습 전 모델이 틀린 단어"] = [i for i, (correct, _, _) in enumerate(before) if not correct]
    if args.train:
        seen = {line.split("\t")[1] for line in args.train.read_text(encoding="utf-8").splitlines() if line}
        groups["  학습에 없던 단어"] = [i for i, (_, truth) in enumerate(rows) if truth not in seen]
    for name, picked in groups.items():
        if picked:
            b, a = accuracy([before[i] for i in picked]), accuracy([after[i] for i in picked])
            print(f"{name} {len(picked):,}단어 — 학습 전: 단어 {b[0]:.1%} · CER {b[1]:.1%} / 학습 후: 단어 {a[0]:.1%} · CER {a[1]:.1%}")


if __name__ == "__main__":
    main()
