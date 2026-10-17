"""AI Hub 053 대용량 손글씨 OCR 데이터 → 인식 모델 학습용 단어 이미지와 라벨 목록.

압축을 풀지 않는다 (다 풀면 252GB, 디스크 여유보다 크다). zip 에서 쪽 PNG 를 바로 읽어 단어 상자만 잘라 저장한다.
사전(학습 코드 사전 = 지금 쓰는 인식 모델 사전, 11,945자)에 없는 글자가 든 단어는 뺀다 — 사전을 바꾸면 RapidOCR 등에
그대로 끼울 수 없다.

  .venv/bin/python -I make_crops.py --labels <라벨 zip> --images <원천 zip>... --out data/val --pages 200
"""

import argparse
import json
import multiprocessing as mp
import random
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_DICT = Path(__file__).resolve().parent / "PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt"
# 저장 높이 상한. 학습 입력은 높이 32~64 이므로 그보다 크게 둘 까닭이 없다 (디스크 절약)
_MAX_HEIGHT = 64
# 상자 둘레 여백 (상자 높이 비율). 손글씨 획이 상자 밖으로 조금 나간다
_PAD = 0.06


def _name(info: zipfile.ZipInfo) -> str:
    # 파일 이름이 CP949 인데 UTF-8 표시가 없어 zipfile 이 cp437 로 읽는다. 이미 제대로 읽힌 이름은 그대로 둔다
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except UnicodeError:
        return info.filename


_images: dict[str, tuple[str, str]] = {}
_zips: dict[str, zipfile.ZipFile] = {}
_charset: set[str] = set()
_out: Path = Path()


def _init(images: dict[str, tuple[str, str]], charset: set[str], out: Path) -> None:
    global _images, _charset, _out
    _images, _charset, _out = images, charset, out


def _read(zip_path: str, member: str) -> bytes:
    if zip_path not in _zips:
        _zips[zip_path] = zipfile.ZipFile(zip_path)
    return _zips[zip_path].read(member)


def _crop_page(job: tuple[str, str, str]) -> tuple[list[str], Counter]:
    labels_zip, member, stem = job
    stats: Counter = Counter()
    if stem not in _images:
        stats["원천 없음"] += 1
        return [], stats
    page = json.loads(_read(labels_zip, member))
    image = cv2.imdecode(np.frombuffer(_read(*_images[stem]), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        stats["원천 읽기 실패"] += 1
        return [], stats
    height, width = image.shape[:2]
    rows = []
    for i, box in enumerate(page["bbox"]):
        text = str(box.get("data", "")).strip()
        if not text:
            stats["글 없음"] += 1
            continue
        if not set(text) <= _charset:
            stats["사전 밖 글자"] += 1
            continue
        x0, x1, y0, y1 = min(box["x"]), max(box["x"]), min(box["y"]), max(box["y"])
        pad = (y1 - y0) * _PAD
        x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        x1, y1 = min(width, int(x1 + pad)), min(height, int(y1 + pad))
        if x1 - x0 < 4 or y1 - y0 < 4:
            stats["상자 너무 작음"] += 1
            continue
        crop = image[y0:y1, x0:x1]
        if crop.shape[0] > _MAX_HEIGHT:
            scale = _MAX_HEIGHT / crop.shape[0]
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), _MAX_HEIGHT), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            stats["저장 실패"] += 1
            continue
        relative = f"{stem[-2:]}/{stem}_{i:03d}.jpg"
        path = _out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
        # 셋째 열: 인식 입력(높이 48)에서의 너비 — 학습 때 비슷한 너비끼리 배치를 묶는다 (PaddleOCR 은 셋째 열을 무시)
        resized = min(320, max(8, -(-48 * crop.shape[1] // crop.shape[0])))
        rows.append(f"{_out.name}/{relative}\t{text}\t{resized}")
        stats["단어"] += 1
    stats["쪽"] += 1
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=0, help="무작위로 고를 쪽 수 (0 = 전부)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    charset = set(_DICT.read_text(encoding="utf-8").splitlines()) | {" "}
    images = {}
    for zip_path in args.images:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".png"):
                    images[Path(_name(info)).stem] = (str(zip_path), info.filename)
    with zipfile.ZipFile(args.labels) as z:
        jobs = [
            (str(args.labels), info.filename, Path(_name(info)).stem)
            for info in z.infolist()
            if info.filename.endswith(".json")
        ]
    random.Random(args.seed).shuffle(jobs)
    if args.pages:
        jobs = jobs[: args.pages]
    args.out.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    rows: list[str] = []
    with mp.Pool(args.workers, initializer=_init, initargs=(images, charset, args.out.resolve())) as pool:
        for n, (page_rows, stats) in enumerate(pool.imap_unordered(_crop_page, jobs, chunksize=8), 1):
            rows += page_rows
            total += stats
            if n % 500 == 0:
                print(f"{n}/{len(jobs)}쪽 · 단어 {total['단어']:,}", flush=True)
    (args.out.parent / f"{args.out.name}_list.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(dict(total))


if __name__ == "__main__":
    main()
