"""출처별 단어 목록을 비율대로 뽑아 학습 목록 하나로. --half 를 주면 필자(쪽) 단위로 A·B 반만 뽑는다 (교차 검증용).

  python3 build_mix.py --total 1500000 --half all --out data/mix-all_list.txt \\
      --source h053:data/h053-ts1_list.txt:36 --source hw91:data/kor-hw-train_list.txt:14 ...

반 나누기 열쇠: 넷째 열(교육 make_id·합성 숫자의 053 쪽)이 있으면 그것, 필기체(kor-hw-train/필자/…)는 필자 폴더,
053 조각은 파일 이름의 쪽 부분(IMG_OCR_53_4PO_10235_052 → IMG_OCR_53_4PO_10235).
같은 필자의 단어가 A·B 에 갈라지지 않아야 반대쪽 반이 '그 필자를 본 적 없는' 시험이 된다.
"""

import argparse
import hashlib
import random
from pathlib import Path


def key(parts: list[str]) -> str:
    if len(parts) >= 4 and parts[3]:
        return parts[3]
    path = Path(parts[0])
    if path.parts[0] == "kor-hw-train" and len(path.parts) > 2:
        return path.parts[1]
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def half_of(text: str) -> str:
    return "A" if int(hashlib.md5(text.encode()).hexdigest(), 16) % 2 == 0 else "B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="이름:목록:비율")
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--half", choices=["all", "A", "B"], default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    specs = [s.split(":") for s in args.source]
    weight_sum = sum(float(w) for _, _, w in specs)
    rows = []
    for name, path, weight in specs:
        lines = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3 or not parts[1]:
                continue
            # 열쇠는 필자 — 같은 필자는 053 단어·합성 숫자 어디서든 같은 반
            if args.half != "all" and half_of(key(parts)) != args.half:
                continue
            lines.append("\t".join(parts[:3]))
        want = round(args.total * float(weight) / weight_sum)
        rng.shuffle(lines)
        picked = (lines * (want // max(1, len(lines)) + 1))[:want] if lines else []
        print(f"{name}: 있는 것 {len(lines):,} → 뽑음 {len(picked):,} ({float(weight) / weight_sum:.0%})", flush=True)
        rows += picked
    rng.shuffle(rows)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"합계 {len(rows):,} → {args.out}")


if __name__ == "__main__":
    main()
