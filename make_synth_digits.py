"""손글씨 숫자·기호 띠 합성 — OCR 이 약한 쉼표·점·하이픈·빗금·괄호를 숫자와 함께 학습시킨다 (문자 중립: 서식 문장 없음).

숫자는 AI Hub 대용량 손글씨(053) 쪽마다 그 필자가 쓴 숫자 단어를 잉크 틈으로 한 자씩 잘라 모은다 (한 띠 = 한 필자).
기호는 데이터에 거의 없어(쉼표 0.04%) 그 필자의 획 굵기로 펜 획을 그린다 — 크기·위치(쉼표는 줄 아래로)·기울기를 흔들고,
쉼표가 앞 숫자에 붙거나 간격이 좁고 넓은 경우를 일부러 넣는다. 출력은 make_crops 와 같은 목록 ('경로\\t글\\t너비').

  python3 make_synth_digits.py --labels TL.zip --images TS1.zip TS2.zip --out data/synth-digits --count 300000
"""

import argparse
import json
import math
import random
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

_NUMBER = re.compile(r"^[\d,.\-/()]+$")
_MAX_HEIGHT = 64
# 숫자 조각을 확인할 인식 모델 — 조각을 한 글자로 읽혀 정답 숫자와 같을 때만 쓴다
_CHECK_MODEL = Path(__file__).resolve().parent / "model/rec_korean_hw.onnx"
_DICT = Path(__file__).resolve().parent / "PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt"


class Ink:
    @staticmethod
    def cut(gray: np.ndarray) -> np.ndarray:
        paper, ink = np.percentile(gray, 90), np.percentile(gray, 3)
        alpha = np.clip((paper - gray.astype(np.float32)) / max(paper - ink, 1.0), 0, 1)
        alpha[alpha < 0.18] = 0
        alpha[(alpha > 0.3).mean(axis=1) > 0.6, :] = 0
        alpha[:, (alpha > 0.3).mean(axis=0) > 0.6] = 0
        return alpha

    @staticmethod
    def split(alpha: np.ndarray, count: int) -> list[np.ndarray] | None:
        ink = (alpha > 0.3).sum(axis=0) > 0
        runs, start = [], None
        for x, on in enumerate([*ink, False]):
            if on and start is None:
                start = x
            elif not on and start is not None:
                runs.append((start, x))
                start = None
        runs = [(a, b) for a, b in runs if b - a > 2]
        if len(runs) != count:
            return None
        widths = sorted(b - a for a, b in runs)
        if widths[-1] > 1.8 * widths[len(widths) // 2]:
            return None
        pieces = [Ink.clean(alpha[:, max(0, a - 2) : b + 2]) for a, b in runs]
        if any((p > 0.3).mean() > 0.5 for p in pieces):
            return None
        return pieces

    @staticmethod
    def clean(piece: np.ndarray) -> np.ndarray:
        """조각에 붙어 온 부스러기(옆 쉼표·점, 칸 선 끝)를 지운다 — 가장 큰 획의 12% 미만인 떨어진 잉크."""
        mask = (piece > 0.3).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 2:
            return piece
        areas = stats[1:, cv2.CC_STAT_AREA]
        height, width = piece.shape
        tops = stats[1:, cv2.CC_STAT_TOP]
        narrow = stats[1:, cv2.CC_STAT_WIDTH] < 0.4 * width
        # 바닥에 붙은 작고 좁은 덩어리는 옆 글자의 점·쉼표 — 12% 를 넘어도 지운다 (2026-11-19: '2023' 띠에 점이 끼어 '2.02.3' 으로 보였다)
        low_dot = (tops >= 0.7 * height) & narrow & (areas < 0.35 * areas.max())
        keep = np.isin(labels, 1 + np.flatnonzero((areas >= 0.12 * areas.max()) & ~low_dot))
        return np.where(cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, piece, 0)

    @staticmethod
    def stroke(alpha: np.ndarray) -> float:
        """획 굵기(px) — 잉크 면적 / 뼈대 길이 근사."""
        mask = (alpha > 0.4).astype(np.uint8)
        area = mask.sum()
        if area == 0:
            return 2.0
        edges = cv2.Canny(mask * 255, 50, 150).sum() / 255
        return float(np.clip(2 * area / max(edges, 1), 1.5, 8))


class Checker:
    """숫자 조각 확인 — 잉크 틈으로 나눈 조각은 짝이 어긋나거나(붙여 쓴 숫자) 서식 칸 선이 섞일 수 있다.
    지금 인식 모델이 그 조각을 정답 숫자 하나로 읽을 때만 쓴다 (틀린 정답 이미지는 학습을 망친다)."""

    def __init__(self) -> None:
        import onnxruntime as ort


        self._session = ort.InferenceSession(str(_CHECK_MODEL), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        # convert_check.CHARS 와 같다 — 그 모듈은 torch 를 불러와 여기서는 쓰지 않는다
        self._chars = ["", *_DICT.read_text(encoding="utf-8").splitlines(), " "]

    def reads(self, alpha: np.ndarray, digit: str) -> bool:
        pad = max(4, alpha.shape[0] // 4)
        ink = np.pad(alpha, pad)
        image = np.clip(255 - ink * 230, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        resized = cv2.resize(image, (max(8, math.ceil(48 * w / h)), 48)).astype(np.float32) / 255.0
        tensor = ((resized - 0.5) / 0.5).transpose(2, 0, 1)[None]
        probs = self._session.run(None, {self._input: tensor})[0][0]
        out, previous = [], 0
        for index in probs.argmax(-1):
            if index != previous and index != 0:
                out.append(self._chars[index])
            previous = index
        return "".join(out) == digit


class Hand:
    """한 필자(쪽)의 숫자 조각과 획 굵기."""

    def __init__(self, glyphs: dict[str, list[np.ndarray]]) -> None:
        self.glyphs = glyphs
        heights = [g.shape[0] for gs in glyphs.values() for g in gs]
        self.height = float(np.median(heights))
        self.stroke = float(np.median([Ink.stroke(g) for gs in glyphs.values() for g in gs[:3]]))

    @classmethod
    def from_page(cls, label: dict, page: np.ndarray, checker: Checker) -> "Hand | None":
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        glyphs: dict[str, list[np.ndarray]] = {}
        for box in label.get("bbox", []):
            text = str(box.get("data", "")).strip()
            if not (_NUMBER.fullmatch(text) and re.search(r"\d", text)):
                continue
            x0, x1, y0, y1 = min(box["x"]), max(box["x"]), min(box["y"]), max(box["y"])
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            alpha = Ink.cut(gray[y0:y1, x0:x1])
            for char, piece in zip(text, Ink.split(alpha, len(text)) or [], strict=False):
                if char.isdigit() and checker.reads(piece, char):
                    glyphs.setdefault(char, []).append(piece)
        return cls(glyphs) if len(glyphs) >= 8 else None


class Text:
    """서류에 흔한 숫자 표기 — 서식 문장은 넣지 않는다 (OCR 은 문자 중립)."""

    # 원 기호의 정답 글자 — 기본 사전에는 ₩ 가 없어 W (인쇄된 ₩ 도 W 로 읽는다). ₩ 를 더한 사전(OCR_DICT)이면 ₩
    WON = "W"
    # 손으로 쓰는 기호(% + = # ² ~ 〃)가 든 표기의 비율 (--symbols)
    SYMBOLS = 0.0

    @staticmethod
    def symbol(rng: random.Random) -> str:
        """사람이 서류에 손으로 쓰는 기호가 든 표기 — 이율·면적·기간·계산·번호·같음표."""
        n = lambda a, b: rng.randint(a, b)  # noqa: E731
        kind = rng.random()
        if kind < 0.22:  # 이율·지분
            return rng.choice([f"{n(1, 99)}%", f"{n(1, 9)}.{n(0, 9)}%", f"{n(1, 20)}.{n(0, 99):02d}%"])
        if kind < 0.42:  # 면적 — ㎡ 는 손으로 m² 로 쓴다
            return rng.choice([f"{n(10, 300)}.{n(0, 99):02d}m²", f"{n(10, 300)}m²", f"{n(10, 300)}.{n(1, 9)}m²"])
        if kind < 0.62:  # 기간
            y = n(2000, 2035)
            return rng.choice([f"{y}.{n(1, 12)}.{n(1, 28)}~{y + n(1, 3)}.{n(1, 12)}.{n(1, 28)}",
                               f"{n(1, 12)}~{n(1, 12)}", f"{y}~{y + n(1, 5)}", f"{n(1, 12)}.{n(1, 28)}~{n(1, 12)}.{n(1, 28)}"])
        if kind < 0.77:  # 계산
            a, b = n(1, 9999), n(1, 9999)
            return rng.choice([f"{a}+{b}={a + b}", f"{a}+{b}", f"{a:,}+{b:,}", f"={a:,}"])
        if kind < 0.9:  # 번호
            return rng.choice([f"#{n(1, 999)}", f"#{n(1, 9999)}", f"{n(1, 99)}-#{n(1, 99)}"])
        return rng.choice(["〃", "〃", f"〃{n(1, 99)}"])  # 같음표

    @staticmethod
    def draw(rng: random.Random) -> str:
        n = lambda a, b: rng.randint(a, b)  # noqa: E731
        if rng.random() < Text.SYMBOLS:
            return Text.symbol(rng)
        kind = rng.random()
        if kind < 0.35:  # 금액
            value = rng.choice([n(1, 999) * 10**k for k in range(2, 7)])
            s = f"{value:,}" if rng.random() < 0.8 else str(value)
            # 원 기호 — 손글씨 ₩ 를 7 로 읽던 것을 막는다 (2026-11-20 확인)
            won = Text.WON
            return rng.choice([s, s, f"({s})", f"*{s}", f"{s}-", f"{won}{s}", f"{won}{s}", f"({won}{s})"])
        if kind < 0.6:  # 날짜
            y, m, d = n(1990, 2035), n(1, 12), n(1, 28)
            sep = rng.choice([".", ".", "-", "/"])
            year = str(y) if rng.random() < 0.75 else f"{y % 100:02d}"
            mm, dd = (f"{m:02d}", f"{d:02d}") if rng.random() < 0.4 else (str(m), str(d))
            tail = "." if sep == "." and rng.random() < 0.4 else ""
            return f"{year}{sep}{mm}{sep}{dd}{tail}"
        if kind < 0.75:  # 전화
            return rng.choice([f"010-{n(1000, 9999)}-{n(1000, 9999)}", f"0{n(2, 64)}-{n(100, 9999)}-{n(1000, 9999)}"])
        if kind < 0.85:  # 주민등록번호 앞자리
            head = f"{n(40, 99):02d}{n(1, 12):02d}{n(1, 28):02d}"
            return rng.choice([f"{head}-{n(1, 4)}******", f"{head}-{n(1000000, 4999999)}", head])
        if kind < 0.93:  # 쪽 표기·괄호 숫자
            k = n(1, 9)
            return rng.choice([f"{k}/{n(k, 9)}", f"({n(1, 99)})", f"{n(1, 99)}.", f"{n(1, 9)}:{n(10, 59)}"])
        return f"{n(1, 9999):,}.{n(1, 99)}"  # 소수


class Pen:
    """기호를 그 필자의 획 굵기로 그린다 — 높이는 숫자 높이에 맞춘다."""

    @staticmethod
    def mark(char: str, height: int, stroke: float, rng: random.Random) -> np.ndarray:
        w = {"," : 0.35, ".": 0.3, "-": 0.55, "/": 0.5, "(": 0.35, ")": 0.35, "*": 0.5, "~": 0.6, ":": 0.3, "W": 0.8, "₩": 0.8,
             "〃": 0.4, "%": 0.7, "+": 0.55, "=": 0.55, "#": 0.6, "m": 0.75}[char]
        width = max(4, int(height * w * rng.uniform(0.8, 1.25)))
        canvas = np.zeros((height, width), np.float32)
        t = max(1, int(round(stroke * rng.uniform(0.8, 1.2))))
        j = lambda v: v + rng.uniform(-0.06, 0.06)  # noqa: E731
        pt = lambda x, y: (int(j(x) * width), int(j(y) * height))  # noqa: E731
        if char == ",":
            cv2.line(canvas, pt(0.55, 0.78), pt(0.35, 1.02), 1.0, t)
            cv2.circle(canvas, pt(0.55, 0.8), max(1, t // 2 + 1), 1.0, -1)
        elif char == ".":
            cv2.circle(canvas, pt(0.5, 0.88), max(1, int(t * rng.uniform(0.6, 1.1))), 1.0, -1)
        elif char == "-":
            cv2.line(canvas, pt(0.1, 0.55), pt(0.9, 0.52), 1.0, t)
        elif char == "/":
            cv2.line(canvas, pt(0.8, 0.1), pt(0.2, 0.95), 1.0, t)
        elif char in "()":
            # 호의 중심을 반대쪽 바깥에 두고 120~240도(왼쪽 괄호) 또는 -60~60도(오른쪽 괄호)만 그린다
            cx = int(width * 1.1) if char == "(" else int(-width * 0.1)
            start, end = (125, 235) if char == "(" else (-55, 55)
            axes = (int(width * 0.95), int(height * 0.55))
            cv2.ellipse(canvas, (cx, height // 2), axes, 0, start, end, 1.0, t)
        elif char == "*":
            c = (width // 2, int(height * 0.35))
            for angle in (90, 30, 150):
                dx, dy = int(math.cos(math.radians(angle)) * width * 0.4), int(math.sin(math.radians(angle)) * height * 0.2)
                cv2.line(canvas, (c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy), 1.0, max(1, t - 1))
        elif char == "~":
            xs = np.linspace(0.1, 0.9, 12)
            points = np.array([pt(x, 0.5 + 0.1 * math.sin(x * 2 * math.pi)) for x in xs], np.int32)
            cv2.polylines(canvas, [points], False, 1.0, t)
        elif char in "W₩":
            # 손글씨 ₩ — W 에 가로획 하나 또는 둘
            points = np.array([pt(0.05, 0.12), pt(0.28, 0.92), pt(0.5, 0.35), pt(0.72, 0.92), pt(0.95, 0.12)], np.int32)
            cv2.polylines(canvas, [points], False, 1.0, t)
            for y in (0.45,) if rng.random() < 0.5 else (0.4, 0.6):
                cv2.line(canvas, pt(0.0, y), pt(1.0, y), 1.0, t)
        elif char == "〃":
            # 같음표 — 위쪽에 비스듬한 짧은 획 둘
            for x in (0.3, 0.7):
                cv2.line(canvas, pt(x, 0.15), pt(x - 0.15, 0.45), 1.0, t)
        elif char == "%":
            r = max(2, int(min(width, height) * 0.14))
            cv2.circle(canvas, pt(0.25, 0.3), r, 1.0, max(1, t - 1))
            cv2.circle(canvas, pt(0.75, 0.75), r, 1.0, max(1, t - 1))
            cv2.line(canvas, pt(0.85, 0.1), pt(0.15, 0.95), 1.0, t)
        elif char == "+":
            cv2.line(canvas, pt(0.1, 0.55), pt(0.9, 0.55), 1.0, t)
            cv2.line(canvas, pt(0.5, 0.2), pt(0.5, 0.9), 1.0, t)
        elif char == "=":
            for y in (0.42, 0.68):
                cv2.line(canvas, pt(0.1, y), pt(0.9, y), 1.0, t)
        elif char == "#":
            for x in (0.38, 0.68):
                cv2.line(canvas, pt(x + 0.08, 0.15), pt(x - 0.08, 0.95), 1.0, t)
            for y in (0.4, 0.7):
                cv2.line(canvas, pt(0.05, y), pt(0.95, y), 1.0, t)
        elif char == "m":
            # 소문자 m — 세로획 하나에 언덕 둘 (숫자 높이의 아래 60%)
            points = [pt(0.08, 0.95), pt(0.08, 0.45), pt(0.3, 0.38), pt(0.45, 0.5), pt(0.45, 0.95)]
            cv2.polylines(canvas, [np.array(points, np.int32)], False, 1.0, t)
            points = [pt(0.45, 0.5), pt(0.68, 0.38), pt(0.88, 0.5), pt(0.88, 0.95)]
            cv2.polylines(canvas, [np.array(points, np.int32)], False, 1.0, t)
        elif char == ":":
            for y in (0.45, 0.85):
                cv2.circle(canvas, pt(0.5, y), max(1, t // 2 + 1), 1.0, -1)
        return cv2.GaussianBlur(canvas, (3, 3), 0)


class Strip:
    @staticmethod
    def render(text: str, hand: Hand, rng: random.Random) -> np.ndarray | None:
        height = int(hand.height)
        pieces = []
        for char in text:
            if char == "²":
                # 위첨자 2 — 그 필자의 2 를 반 크기로 위쪽에
                if "2" not in hand.glyphs:
                    return None
                small = rng.choice(hand.glyphs["2"])
                sh = max(2, int(height * rng.uniform(0.42, 0.55)))
                small = cv2.resize(small, (max(2, int(small.shape[1] * sh / small.shape[0])), sh))
                g = np.zeros((height, small.shape[1]), np.float32)
                g[: small.shape[0]] = small
            elif char.isdigit():
                if char not in hand.glyphs:
                    return None
                g = rng.choice(hand.glyphs[char])
                scale = height / g.shape[0] * rng.uniform(0.9, 1.08)
                g = cv2.resize(g, (max(2, int(g.shape[1] * scale)), max(2, int(g.shape[0] * scale))))
            else:
                g = Pen.mark(char, height, hand.stroke, rng)
            pieces.append((char, g))
        gaps = []
        for i, (char, _) in enumerate(pieces[1:], 1):
            # 쉼표·점은 앞 숫자에 붙기 쉽다 — 겹치거나 좁게
            tight = char in ",." or pieces[i - 1][0] in ",."
            gaps.append(int(height * (rng.uniform(-0.12, 0.08) if tight else rng.uniform(0.0, 0.3))))
        canvas_h = int(height * 1.35)
        width = sum(g.shape[1] for _, g in pieces) + sum(max(0, x) for x in gaps) + 8
        canvas = np.zeros((canvas_h, width), np.float32)
        x = 4
        top0 = int(height * 0.12)
        for i, (_, g) in enumerate(pieces):
            gh, gw = g.shape
            top = min(canvas_h - gh, max(0, top0 + int(height * rng.uniform(-0.06, 0.06))))
            right = min(width, x + gw)
            canvas[top : top + gh, x:right] = np.maximum(canvas[top : top + gh, x:right], g[:, : right - x])
            x += gw + (gaps[i] if i < len(gaps) else 0)
            x = max(0, x)
        canvas = canvas[:, : max(8, x + 4)]
        # 기울기(손잡이) — 전단
        shear = rng.uniform(-0.25, 0.05) if rng.random() < 0.1 else rng.uniform(-0.05, 0.3)
        h, w = canvas.shape
        matrix = np.float32([[1, -shear, max(0, shear) * h], [0, 1, 0]])
        canvas = cv2.warpAffine(canvas, matrix, (w + int(abs(shear) * h), h))
        # 종이·잉크
        paper = rng.uniform(200, 250)
        ink = np.array(rng.choice([(30, 30, 30), (70, 30, 20), (110, 60, 40), (90, 90, 90)]), np.float32)
        bg = np.full((*canvas.shape, 3), paper, np.float32) + np.random.default_rng(rng.randint(0, 2**31)).normal(0, 4, (*canvas.shape, 3))
        alpha = (canvas * rng.uniform(0.6, 1.0))[..., None]
        image = np.clip(bg * (1 - alpha) + ink * alpha, 0, 255).astype(np.uint8)
        if image.shape[0] > _MAX_HEIGHT:
            scale = _MAX_HEIGHT / image.shape[0]
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), _MAX_HEIGHT), interpolation=cv2.INTER_AREA)
        return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300000)
    parser.add_argument("--pages", type=int, default=4000, help="숫자를 모을 쪽(필자) 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--won", default="W", help="원 기호의 정답 글자 — ₩ 를 더한 사전으로 학습할 때 ₩")
    parser.add_argument("--symbols", type=float, default=0.0, help="손으로 쓰는 기호(% + = # ² ~ 〃)가 든 표기의 비율")
    args = parser.parse_args()
    Text.WON, Text.SYMBOLS = args.won, args.symbols
    rng = random.Random(args.seed)
    labels = zipfile.ZipFile(args.labels, metadata_encoding="cp949")
    images = {}
    for path in args.images:
        z = zipfile.ZipFile(path, metadata_encoding="cp949")
        for name in z.namelist():
            if name.lower().endswith((".png", ".jpg")):
                images[Path(name).stem] = (z, name)
    members = [n for n in labels.namelist() if n.endswith(".json") and Path(n).stem in images]
    rng.shuffle(members)
    hands = []
    checker = Checker()
    for member in members[: args.pages]:
        z, name = images[Path(member).stem]
        page = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        if page is None:
            continue
        hand = Hand.from_page(json.loads(labels.read(member)), page, checker)
        if hand:
            hands.append((Path(member).stem, hand))
    print(f"필자(쪽) {len(hands)} / {min(len(members), args.pages)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    made = 0
    while made < args.count:
        page_id, hand = rng.choice(hands)
        text = Text.draw(rng)
        image = Strip.render(text, hand, rng)
        if image is None:
            continue
        relative = f"{made // 10000:03d}/{made:07d}.jpg"
        path = args.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        path.write_bytes(buf.tobytes())
        resized = min(320, max(8, -(-48 * image.shape[1] // image.shape[0])))
        # 넷째 열 = 필자(쪽) — 반쪽 OCR 나누기에서 그 필자의 053 단어와 같은 반에 넣는다 (build_mix.key)
        rows.append(f"{args.out.name}/{relative}\t{text}\t{resized}\t{page_id}")
        made += 1
        if made % 20000 == 0:
            print(f"{made}/{args.count}", flush=True)
    (args.out.parent / f"{args.out.name}_list.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
