"""손글씨 문장(한국어 글자체 이미지 샘플)으로 학습 전·후 인식 모델을 비교한다 — 학습에 안 쓴 다른 데이터, 흘림 글씨가 많다.

띄어쓰기는 손글씨에서 애매하므로 빼고 CER 을 잰다.
  .venv/bin/python -I eval_sentences.py <한국어 글자체 이미지 샘플 폴더> <학습 후 ONNX> [문장 수]
"""
import json, mmap, random, re, sys
from pathlib import Path

# -I 로 돌리면 스크립트 폴더가 모듈 경로에 없다 (convert_check 를 같은 폴더에서 읽는다)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import onnxruntime as ort
from rapidfuzz.distance import Levenshtein

from convert_check import CHARS, ONNX, decode, preprocess

root, new_model = Path(sys.argv[1]), sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
pattern = re.compile(rb'\{"id": "\d+", "image_id": "(\d+)", "text": "((?:[^"\\]|\\.)*)", "attributes": \{"type": "' + "문장".encode() + rb'"')
labels = {}
with open(root / "라벨링데이터/handwriting_data_info_clean.json", "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
    for match in pattern.finditer(m):
        labels[match.group(1).decode()] = json.loads(b'"' + match.group(2) + b'"')
files = sorted((root / "원천데이터/01_handwriting_sentence_images/1_sentence").glob("*.png"))
random.Random(0).shuffle(files)
files = files[:limit] if limit else files
sessions = {"학습 전": ort.InferenceSession(str(ONNX)), "학습 후": ort.InferenceSession(new_model)}
totals = {name: [0, 0, 0] for name in sessions}
charset, outside, examples = set(CHARS), 0, []
for path in files:
    truth = labels[path.stem].replace(" ", "")
    outside += sum(c not in charset for c in truth)
    x = preprocess(cv2.imread(str(path)))
    texts = {}
    for name, session in sessions.items():
        text = decode(session.run(None, {"x": x})[0][0]).replace(" ", "")
        totals[name][0] += Levenshtein.distance(text, truth)
        totals[name][1] += len(truth)
        totals[name][2] += text == truth
        texts[name] = text
    if len(examples) < 5:
        examples.append((truth, texts))
for name, (edits, length, exact) in totals.items():
    print(f"{name}: CER {edits / length:.1%} · 문장 통째로 맞음 {exact / len(files):.1%} ({len(files)}문장, 띄어쓰기 제외)")
print(f"사전에 없는 글자 {outside}개 (두 모델 모두 못 읽음)")
for truth, texts in examples:
    print(f"\n정답  : {truth}")
    for name, text in texts.items():
        print(f"{name}: {text}")
