"""AI Hub OCR 데이터(교육) → 인식 모델 학습용 단어 이미지 — 서툰 글씨(초·중등) 보강.

053 과 같은 출력(높이 64 이하 JPEG + '경로\\t글\\t너비' 목록)에 넷째 열로 make_id 를 남긴다 (학습·시험을 make_id 로 나눈다).
거름 (2026-10 점검): LaTeX 표기(\\rightarrow·P_{z}), 높이 16px 미만 흐린 조각, 같은 쪽 중복, 사전 밖 글자.
압축은 풀지 않고 zip 에서 바로 읽는다. 라벨 zip 과 원천 zip 의 파일은 이름(확장자 뺀 것)으로 짝짓는다.

  python3 make_crops_edu.py --labels TL_*.zip --images TS_*.zip --out data/edu-train --workers 12
"""

import argparse
import json
import multiprocessing as mp
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_DICT = Path(__file__).resolve().parent / "PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt"
_MAX_HEIGHT = 64
_PAD = 0.06
_MIN_HEIGHT = 16
_LATEX = set("\\_{}^")

_zips: dict[str, zipfile.ZipFile] = {}
_charset: set[str] = set()
_out = Path()


def _init(charset: set[str], out: Path) -> None:
    global _charset, _out
    _charset, _out = charset, out


def _zip(path: str) -> zipfile.ZipFile:
    if path not in _zips:
        _zips[path] = zipfile.ZipFile(path, metadata_encoding="cp949")
    return _zips[path]


def _page(job: tuple[str, str, str, str]) -> tuple[list[str], Counter]:
    labels_zip, member, images_zip, image_member = job
    stats: Counter = Counter()
    label = json.loads(_zip(labels_zip).read(member))
    image = cv2.imdecode(np.frombuffer(_zip(images_zip).read(image_member), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        stats["원천 읽기 실패"] += 1
        return [], stats
    make_id = str(label.get("Images", {}).get("make_id", "?"))
    height, width = image.shape[:2]
    rows, seen = [], set()
    stem = Path(member).stem
    for i, box in enumerate(label.get("Bbox") or label.get("bbox") or []):
        text = str(box.get("data", "")).strip()
        if not text:
            stats["글 없음"] += 1
            continue
        if set(text) & _LATEX:
            stats["LaTeX 표기"] += 1
            continue
        if not set(text) <= _charset:
            stats["사전 밖 글자"] += 1
            continue
        x0, x1, y0, y1 = min(box["x"]), max(box["x"]), min(box["y"]), max(box["y"])
        if y1 - y0 < _MIN_HEIGHT:
            stats["너무 작음"] += 1
            continue
        key = (text, x0 // 8, y0 // 8)
        if key in seen:
            stats["같은 쪽 중복"] += 1
            continue
        seen.add(key)
        pad = (y1 - y0) * _PAD
        x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        x1, y1 = min(width, int(x1 + pad)), min(height, int(y1 + pad))
        crop = image[y0:y1, x0:x1]
        if crop.shape[0] > _MAX_HEIGHT:
            scale = _MAX_HEIGHT / crop.shape[0]
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), _MAX_HEIGHT), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            continue
        relative = f"{stem[-2:]}/{stem}_{i:03d}.jpg"
        path = _out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
        resized = min(320, max(8, -(-48 * crop.shape[1] // crop.shape[0])))
        rows.append(f"{_out.name}/{relative}\t{text}\t{resized}\t{make_id}")
        stats["단어"] += 1
    stats["쪽"] += 1
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    images = {}
    for path in args.images:
        for name in zipfile.ZipFile(path, metadata_encoding="cp949").namelist():
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                images[Path(name).stem] = (str(path), name)
    jobs = []
    for path in args.labels:
        for name in zipfile.ZipFile(path, metadata_encoding="cp949").namelist():
            if name.endswith(".json") and Path(name).stem in images:
                jobs.append((str(path), name, *images[Path(name).stem]))
    print(f"쪽 {len(jobs)}", flush=True)
    charset = set(_DICT.read_text(encoding="utf-8").splitlines()) | {" "}
    args.out.mkdir(parents=True, exist_ok=True)
    rows, total = [], Counter()
    with mp.Pool(args.workers, initializer=_init, initargs=(charset, args.out)) as pool:
        for n, (page_rows, stats) in enumerate(pool.imap_unordered(_page, jobs, chunksize=8), 1):
            rows += page_rows
            total.update(stats)
            if n % 2000 == 0:
                print(f"{n}/{len(jobs)} · 단어 {total['단어']:,}", flush=True)
    (args.out.parent / f"{args.out.name}_list.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(dict(total), flush=True)


if __name__ == "__main__":
    main()
