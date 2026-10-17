"""AI Hub 다양한 형태의 한글 문자 OCR → 인식 모델 학습용 단어 이미지와 라벨 목록.

make_crops.py(053)와 같은 출력(높이 64 이하 JPEG + '경로\\t글\\t너비' 목록)을 낸다. 압축은 풀지 않고 zip 에서 바로 읽는다.
  필기체: 단어 이미지(2.단어)를 글자 상자들의 합 + 여백으로 다시 잘라 053 단어 상자와 맞춘다. 필자 번호가 있어
          학습(001~122)과 검증(138~152) 필자가 겹치지 않는다.
  인쇄체: 정부 서식 50종의 쪽 이미지(300dpi)에서 단어 상자를 자른다. 서식 글자(source=form)는 같은 서식의 쪽마다
          똑같이 반복되므로 (서식, 글)마다 --form-cap 장까지만 쓴다. 채운 값(source=content)은 모두 쓴다.
사전(11,945자)에 없는 글자가 든 단어는 뺀다 — 사전을 바꾸면 기존 모델 자리에 그대로 끼울 수 없다.

  .venv/bin/python -I make_crops_kor.py handwriting --labels <라벨 zip> --images <원천 zip> --out data/kor-hw-train
  .venv/bin/python -I make_crops_kor.py printed --labels <라벨 zip> --images <원천 zip> --out data/kor-print-train \\
      --forms 1-45 --pages 12000 --form-cap 10
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

# -I 로 돌리면 스크립트 폴더가 모듈 경로에 없다 (make_crops 를 같은 폴더에서 읽는다)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from make_crops import _DICT, _MAX_HEIGHT, _PAD

_zips: dict[str, zipfile.ZipFile] = {}
_charset: set[str] = set()
_out: Path = Path()


def _init(charset: set[str], out: Path) -> None:
    global _charset, _out
    _charset, _out = charset, out


def _read(zip_path: str, member: str) -> bytes:
    if zip_path not in _zips:
        _zips[zip_path] = zipfile.ZipFile(zip_path)
    return _zips[zip_path].read(member)


def _save(image: np.ndarray, box: tuple[int, int, int, int], relative: str) -> int | None:
    """상자(x0, y0, x1, y1)를 여백을 붙여 잘라 저장하고 인식 입력 너비를 돌려준다. 너무 작으면 None."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    pad = (y1 - y0) * _PAD
    x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    x1, y1 = min(width, int(x1 + pad)), min(height, int(y1 + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] > _MAX_HEIGHT:
        scale = _MAX_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), _MAX_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    path = _out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    # 셋째 열: 인식 입력(높이 48)에서의 너비 — 학습 때 비슷한 너비끼리 배치를 묶는다
    return min(320, max(8, -(-48 * crop.shape[1] // crop.shape[0])))


def _handwriting(job: tuple[str, str, str]) -> tuple[list[str], Counter]:
    labels_zip, images_zip, member = job
    stats: Counter = Counter()
    label = json.loads(_read(labels_zip, member))
    letters = label["text"]["word"]
    text = "".join(letter["value"] for letter in letters).strip()
    if not text:
        stats["글 없음"] += 1
        return [], stats
    if not set(text) <= _charset:
        stats["사전 밖 글자"] += 1
        return [], stats
    image = cv2.imdecode(np.frombuffer(_read(images_zip, member[:-5] + ".jpg"), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        stats["원천 읽기 실패"] += 1
        return [], stats
    boxes = np.array([letter["charbox"] for letter in letters])
    box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    stem = Path(member).stem
    relative = f"{label['license']['writer_no']}/{stem}.jpg"
    width = _save(image, box, relative)
    if width is None:
        stats["상자 너무 작음"] += 1
        return [], stats
    stats["단어"] += 1
    return [f"{_out.name}/{relative}\t{text}\t{width}"], stats


def _printed(job: tuple[str, str, str, set[int]]) -> tuple[list[str], Counter]:
    """쪽 하나. keep: 이 쪽에서 쓸 서식 글자 단어의 번호 (서식마다 상한을 넘지 않게 미리 골라 둔다)."""
    labels_zip, images_zip, member, keep = job
    stats: Counter = Counter()
    label = json.loads(_read(labels_zip, member))
    image = cv2.imdecode(np.frombuffer(_read(images_zip, member[:-5] + ".jpg"), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        stats["원천 읽기 실패"] += 1
        return [], stats
    stem = Path(member).stem
    form = member.split("/")[1]
    rows = []
    for i, word in enumerate(label["text"]["word"]):
        text = (word.get("value") or "").strip()
        source = word.get("source")
        if source == "form" and i not in keep:
            stats["서식 글자 반복"] += 1
            continue
        if not text:
            stats["글 없음"] += 1
            continue
        if not set(text) <= _charset:
            stats["사전 밖 글자"] += 1
            continue
        relative = f"{source}/{form}/{stem}_{i:03d}.jpg"
        width = _save(image, tuple(word["wordbox"]), relative)
        if width is None:
            stats["상자 너무 작음"] += 1
            continue
        rows.append(f"{_out.name}/{relative}\t{text}\t{width}")
        stats[f"단어({source})"] += 1
    stats["쪽"] += 1
    return rows, stats


def _forms(spec: str) -> set[str]:
    low, _, high = spec.partition("-")
    return {f"{n:03d}" for n in range(int(low), int(high or low) + 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["handwriting", "printed"])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--words", type=int, default=0, help="필기체: 무작위로 고를 단어 수 (0 = 전부)")
    parser.add_argument("--forms", default="1-50", help="인쇄체: 쓸 서식 번호 범위 (예: 1-45)")
    parser.add_argument("--pages", type=int, default=0, help="인쇄체: 무작위로 고를 쪽 수 (0 = 전부)")
    parser.add_argument("--form-cap", type=int, default=0, help="인쇄체: (서식, 글)마다 쓸 서식 글자 단어 수 (0 = 전부)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    charset = set(_DICT.read_text(encoding="utf-8").splitlines()) | {" "}
    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.labels) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".json"))
    if args.kind == "handwriting":
        members = [m for m in members if m.startswith("2.단어/")]
        rng.shuffle(members)
        jobs: list[tuple] = [(str(args.labels), str(args.images), m) for m in members[: args.words or None]]
        work: Callable[[tuple], tuple[list[str], Counter]] = _handwriting
    else:
        forms = _forms(args.forms)
        members = [m for m in members if m.split("/")[1] in forms]
        rng.shuffle(members)
        members = members[: args.pages or None]
        # 서식 글자 상한: (서식, 글)마다 처음 만난 form_cap 개만 남긴다 — 쪽 순서는 이미 섞었다
        seen: Counter = Counter()
        jobs = []
        with zipfile.ZipFile(args.labels) as z:
            for m in members:
                words = json.loads(z.read(m))["text"]["word"]
                keep: set[int] = set()
                for i, word in enumerate(words):
                    if word.get("source") != "form":
                        continue
                    key = (m.split("/")[1], word.get("value"))
                    if not args.form_cap or seen[key] < args.form_cap:
                        seen[key] += 1
                        keep.add(i)
                jobs.append((str(args.labels), str(args.images), m, keep))
        work = _printed
    args.out.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    rows: list[str] = []
    with mp.Pool(args.workers, initializer=_init, initargs=(charset, args.out.resolve())) as pool:
        for n, (job_rows, stats) in enumerate(pool.imap_unordered(work, jobs, chunksize=16), 1):
            rows += job_rows
            total += stats
            if n % 5000 == 0:
                print(f"{n}/{len(jobs)} · 단어 {len(rows):,}", flush=True)
    (args.out.parent / f"{args.out.name}_list.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(dict(total))


if __name__ == "__main__":
    main()
