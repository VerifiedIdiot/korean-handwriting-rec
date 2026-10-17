"""인식 모델(ONNX) 여러 개를 단어 목록 여러 개로 잰다 — 목록마다 단어 정확도·CER 와 갈래별(인쇄체 서식 글자/채운 값,
빗금·숫자·그 밖) 결과. 학습 전·학습 모델과 새 모델을 같은 단어로 비교한다.

  .venv/bin/python -I eval_lists.py --model 원래=<onnx> --model 학습=<onnx> --list data/kor-print-val_list.txt --words 5000
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import onnxruntime as ort
from rapidfuzz.distance import Levenshtein

from convert_check import decode, preprocess


def groups(relative: str, text: str) -> list[str]:
    """make_crops_kor 인쇄체 경로(<목록>/form|content/…)면 서식 글자·채운 값으로도 가른다."""
    parts = relative.split("/")
    out = ["전체"]
    if len(parts) > 2 and parts[1] in ("form", "content"):
        out.append("서식 글자" if parts[1] == "form" else "채운 값")
    out.append("빗금" if "/" in text else "숫자" if any(c.isdigit() for c in text) else "그 밖")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="이름=ONNX 경로")
    parser.add_argument("--list", action="append", required=True, type=Path)
    parser.add_argument("--words", type=int, default=5000, help="목록마다 무작위로 고를 단어 수 (0 = 전부)")
    args = parser.parse_args()

    models = {name: ort.InferenceSession(path) for name, path in (m.split("=", 1) for m in args.model)}
    for list_path in args.list:
        rows = [line.split("\t") for line in list_path.read_text(encoding="utf-8").splitlines() if line]
        random.Random(0).shuffle(rows)
        rows = rows[: args.words or None]
        # (갈래, 모델) → [맞음, 편집 거리, 글자 수, 단어 수]
        score: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for relative, text, *_ in rows:
            x = preprocess(cv2.imread(str(list_path.parent / relative)))
            for name, session in models.items():
                out = decode(session.run(None, {"x": x})[0][0])
                for group in groups(relative, text):
                    s = score[(group, name)]
                    s[0] += out == text
                    s[1] += Levenshtein.distance(out, text)
                    s[2] += len(text)
                    s[3] += 1
        print(f"== {list_path.name} ({len(rows):,}단어)")
        for group in dict.fromkeys(g for g, _ in score):
            n = score[(group, next(iter(models)))][3]
            cells = " · ".join(
                f"{name} {s[0] / s[3]:.1%} (CER {s[1] / max(s[2], 1):.1%})" for name in models for s in [score[(group, name)]]
            )
            print(f"  {group} {n:,}단어 — {cells}", flush=True)


if __name__ == "__main__":
    main()
