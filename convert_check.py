"""PaddleOCR 사전학습 가중치(korean PP-OCRv5 mobile rec) → PyTorch. 지금 쓰는 ONNX 와 같은 답을 내는지 대조하고,
ONNX 의 손글씨 단어 정확도(학습 전 기준선)를 잰다.

  .venv/bin/python -I convert_check.py <라벨 목록> [대조할 단어 수]
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "PaddleOCR2Pytorch"))
from converter.ppocr_v5_rec_converter import PPOCRv5RecConverter, read_network_config_from_yaml  # noqa: E402

DICT = ROOT / "PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt"
# 기존 모델: huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx 의 inference.onnx 를 이 이름으로 받아 둔다
ONNX = ROOT / "pretrained/rec_korean.onnx"
PRETRAINED = ROOT / "pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
# CTC 출력 순서: 0 = blank, 사전 글자, 마지막 = 공백 (use_space_char)
CHARS = ["", *DICT.read_text(encoding="utf-8").splitlines(), " "]


def network_config() -> dict:
    cfg = yaml.safe_load((ROOT / "PaddleOCR2Pytorch/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml").read_text(encoding="utf-8"))
    cfg["Global"]["character_dict_path"] = str(DICT)
    path = ROOT / "configs/torch_korean_rec.yml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return read_network_config_from_yaml(str(path))


def load_torch() -> torch.nn.Module:
    return PPOCRv5RecConverter(network_config(), str(PRETRAINED)).net.eval()


def preprocess(image: np.ndarray) -> np.ndarray:
    """RapidOCR·PaddleOCR 인식 입력과 같게: 높이 48, 비율 유지, (x/255-0.5)/0.5, BGR, CHW."""
    height, width = image.shape[:2]
    resized = cv2.resize(image, (max(8, math.ceil(48 * width / height)), 48)).astype(np.float32) / 255.0
    return ((resized - 0.5) / 0.5).transpose(2, 0, 1)[None]


def decode(probs: np.ndarray) -> str:
    out, previous = [], 0
    for index in probs.argmax(-1):
        if index != previous and index != 0:
            out.append(CHARS[index])
        previous = index
    return "".join(out)


def main() -> None:
    rows = [line.split("\t", 1) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line]
    compare = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    session = ort.InferenceSession(str(ONNX))
    net = load_torch()
    max_diff, same, correct, edits, length = 0.0, 0, 0, 0, 0
    for n, (relative, truth) in enumerate(rows):
        x = preprocess(cv2.imread(str(Path(sys.argv[1]).parent / relative)))
        p_onnx = session.run(None, {"x": x})[0][0]
        text = decode(p_onnx)
        correct += text == truth
        edits += Levenshtein.distance(text, truth)
        length += len(truth)
        if n < compare:
            with torch.no_grad():
                p_torch = net(torch.from_numpy(x))[0].numpy()
            max_diff = max(max_diff, float(np.abs(p_torch - p_onnx).max()))
            same += decode(p_torch) == text
    print(f"PyTorch 대 ONNX ({compare}단어): 확률 최대 차이 {max_diff:.2e}, 읽은 글이 같은 단어 {same}/{compare}")
    print(f"학습 전 ONNX 손글씨 정확도 ({len(rows)}단어): 단어 {correct / len(rows):.1%}, CER {edits / length:.1%}")


if __name__ == "__main__":
    main()
