# korean-handwriting-rec

**손글씨도 읽는 PP-OCRv5.**

한국어 | [English](#english)

> 제조, 운송, 의료, 민원 업무처럼 컴퓨터 사용이 적거나 서면 기록이 중요한 현장에서는 지금도 서식에 손으로 적은 문서가 오갑니다.
>
> 이런 문서는 스캔본이나 휴대폰 사진처럼 상태가 고르지 않은 이미지로 들어오고, 그 안의 거친 손글씨를 읽어 내야 합니다.
>
> 하지만 기존 OCR 모델은 인쇄체 위주로 학습되어 이런 글씨를 제대로 읽지 못합니다. 이 문제를 풀려고 만들어 공개합니다.

korean-handwriting-rec은 한국어 손글씨를 읽는 OCR 인식 모델입니다. PP-OCRv5 한국어 인식 모델을 AI Hub 손글씨·공문서 데이터 460만 단어로 파인튜닝했습니다. 이미 PP-OCRv5를 쓰고 있다면 모델 파일만 바꾸면 됩니다. 모델은 [Hugging Face](https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec)에서도 받을 수 있습니다.

- **손글씨 단어 정확도 97.4%**: 기존 모델(67.2%)보다 30%p 높습니다.
- **처음 보는 글씨체도 99.1%**: 학습에 없던 필자 15명의 손글씨로 검증했습니다.
- **공문서 인쇄 글자 99.2%**: 학습하지 않은 정부 서식 5종에서도 기존 모델(88.7%)보다 정확합니다.
- **파일 하나만 교체**: 입출력과 글자 사전이 기존 모델과 같아 RapidOCR 등 PP-OCRv5 기반 도구에 그대로 끼워 씁니다.
- **15MB, CPU로 충분**: GPU도 외부 API도 필요 없어 사내 서버에서 돌립니다. 생성형 모델처럼 없는 값을 지어내지 않고, 이미지에 있는 글자만 읽습니다.

## 빠른 시작

RapidOCR 3.x에서 인식 모델과 사전 경로만 바꿉니다.

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Rec.model_path": "model/rec_korean_hw.onnx",
    "Rec.rec_keys_path": "model/ppocrv5_korean_dict.txt",
})
result = engine("scan.png")
```

## AS-IS와 TO-BE

AS-IS는 기존 PP-OCRv5 한국어 인식 모델(PaddleOCR `korean_PP-OCRv5_mobile_rec`), TO-BE는 이 모델입니다. 모두 학습에 쓰지 않은 평가 세트로 쟀고, 단어 정확도는 단어 전체가 맞아야 정답으로 칩니다.

| 평가 세트 | AS-IS | TO-BE |
|---|---|---|
| **손글씨 단어 2만 개** (대용량 손글씨 OCR 데이터) | 67.2% | **97.4%** |
| 그중 종이에 쓴 서식 | 66.8% | 98.1% |
| 그중 종이에 쓴 자유 글씨 | 73.9% | 96.9% |
| 그중 태블릿에 쓴 서식 | 55.8% | 97.1% |
| 그중 태블릿에 쓴 자유 글씨 | 55.0% | 93.8% |
| 그중 학습에 없던 단어 | 66.7% | 96.3% |
| 그중 AS-IS가 틀린 단어 6,550개 | 0% | 92.9% |
| **학습에 없던 필자의 손글씨 6천 단어** (다양한 형태의 한글 문자 OCR) | 56.7% | **99.1%** |
| **학습에 없던 정부 서식 5종의 인쇄 글자 6천 단어** (다양한 형태의 한글 문자 OCR) | 88.7% | **99.2%** |
| 그중 숫자가 든 단어 | 95.7% | 99.3% |
| 그중 서식에 미리 인쇄된 항목명 | 92.9% | 97.8% |
| 그중 서식에 기재한 값 | 88.4% | 99.3% |
| **흘림체 문장 1,300개의 CER**, 낮을수록 좋음 (한국어 글자체 이미지) | 30.0% | **7.8%** |

같은 이미지를 두 모델에 읽힌 결과입니다.

| 종류 | 정답 | AS-IS | TO-BE |
|---|---|---|---|
| 손글씨 | 체납처분유예 | 제납처분예 | 체납처분유예 |
| 손글씨 | 등록면허세 | 등록면해세 | 등록면허세 |
| 손글씨 금액 | 3235원 | 2 | 3235원 |
| 처음 보는 필자의 손글씨 | 이력서 | 이 령서 | 이력서 |
| 처음 보는 필자의 손글씨 | 환율 | 화을 | 환율 |
| 서식 인쇄 금액 | 59,470,000 | 59.470.000 | 59,470,000 |
| 서식 인쇄 날짜 | 2017.10.28 | 201710.28 | 2017.10.28 |

## 자주 묻는 질문

**한국어 손글씨를 읽는 OCR 모델을 찾고 있습니다. 바로 쓸 수 있나요?**
네. 서식에 손으로 적은 한국어 단어를 읽도록 학습했고, 학습에 쓰지 않은 손글씨 2만 단어에서 단어 정확도 97.4%를 냅니다. 빠른 시작처럼 RapidOCR에서 모델 파일만 바꾸면 됩니다.

**지금 쓰는 PP-OCRv5 파이프라인에 넣을 수 있나요?**
네. 입출력과 글자 사전(11,945자)이 기존 PP-OCRv5 한국어 인식 모델과 같아서 인식 모델 파일과 사전 경로만 바꾸면 됩니다. 글자 검출 모델은 쓰던 것을 그대로 씁니다.

**인쇄된 글자도 읽나요?**
네. 공문서 인쇄 글자를 함께 학습해, 학습하지 않은 정부 서식 5종의 인쇄 글자에서 99.2%로 기존 모델(88.7%)보다 정확합니다.

**GPU가 있어야 하나요?**
아니요. 15MB ONNX 모델이라 CPU로 충분하고, 서류를 외부 API로 보내지 않고 사내 서버에서 처리할 수 있습니다.

**라이선스는 무엇인가요?**
모델과 코드 모두 Apache-2.0입니다.

## 학습 방법

PaddleOCR의 사전학습 모델을 PyTorch로 옮겨 Mac GPU(MPS)에서 두 단계로 파인튜닝했습니다.

1. **손글씨**: AI Hub 대용량 손글씨 OCR 데이터 386만 단어를 1 epoch 학습했습니다. 인쇄 글자 인식도 함께 지키도록 샘플의 20%는 같은 단어를 인쇄체로 렌더링해 섞었습니다.
2. **공문서 인쇄**: 서류의 인쇄 글자까지 함께 읽도록 정부 서식 인쇄 42만 단어와 필기체 30만 단어에 1단계 손글씨 120만 단어를 섞어 이어서 학습했습니다. 손글씨 정확도는 지키면서 처음 보는 서식의 인쇄 글자를 99.2%까지 읽습니다.

글자 사전(11,945자)은 기존 모델 것을 그대로 써서 호환성을 지켰습니다.

## 데이터 출처와 라이선스

학습 데이터는 AI Hub에서 신청해 받을 수 있습니다.

이 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "[대용량 손글씨 OCR 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=605)"와 "[다양한 형태의 한글 문자 OCR](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=91)"을 활용하여 학습했고, "[한국어 글자체 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=81)"로 평가했습니다. 이 데이터는 AI 허브(aihub.or.kr)에서 다운로드할 수 있습니다.

This model was trained using datasets from The Open AI Dataset Project (AI-Hub, S. Korea), "대용량 손글씨 OCR 데이터" and "다양한 형태의 한글 문자 OCR", and evaluated with "한국어 글자체 이미지". All data information can be accessed through AI-Hub (www.aihub.or.kr).

기반 모델과 글자 사전은 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), PyTorch 변환은 [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)를 사용했습니다(둘 다 Apache-2.0). 이 저장소도 Apache-2.0 라이선스입니다.

## English

**korean-handwriting-rec** is a Korean handwriting OCR model. It is the PP-OCRv5 Korean recognition model (`korean_PP-OCRv5_mobile_rec`) fine-tuned on 4.6M Korean word images of handwriting and printed government forms from AI Hub. Input, output and the 11,945-character dictionary are unchanged, so any PP-OCRv5 ONNX pipeline such as RapidOCR can use it by swapping one model file (see the quick start above).

| Held-out evaluation set | PP-OCRv5 Korean | korean-handwriting-rec |
|---|---|---|
| Handwritten words (20K) | 67.2% | **97.4%** |
| Handwriting by 15 unseen writers (6K words) | 56.7% | **99.1%** |
| Printed text on 5 unseen government forms (6K words) | 88.7% | **99.2%** |
| Cursive sentences, CER (1.3K, lower is better) | 30.0% | **7.8%** |

The model is a 15 MB ONNX file that runs on CPU without an external API. It reads only the text in the image and does not generate values the way generative models can. Model and code are licensed under Apache-2.0.
