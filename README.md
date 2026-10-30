# korean-handwriting-rec · 한국어 손글씨 OCR

**손글씨도 읽는 PP-OCRv5.**

한국어 | [English](README.en.md) · [모델 다운로드](https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec)

korean-handwriting-rec은 PP-OCRv5에 한국어 손글씨를 더 가르친 OCR 인식 모델입니다. AI Hub 손글씨·공문서 데이터로 추가 학습했으며, 약 15MB의 ONNX 모델을 CPU에서 실행할 수 있습니다.

**손글씨 단어 정확도 67.2% → 97.4%.** 학습에 쓰지 않은 2만 단어에서 기존 모델과 비교했습니다. [평가 결과](#인식-성능)

RapidOCR에서는 인식 모델과 글자 사전 경로를 지정해 연결합니다.

## 빠른 시작

저장소를 받은 뒤 RapidOCR 3.x와 ONNX Runtime을 설치합니다.

```bash
git clone https://github.com/VerifiedIdiot/korean-handwriting-rec.git
cd korean-handwriting-rec
python -m pip install "rapidocr>=3,<4" onnxruntime
```

저장소 루트에서 실행합니다. `scan.png`는 읽을 이미지의 경로로 바꿉니다.

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Rec.model_path": "model/rec_korean_hw.onnx",
    "Rec.rec_keys_path": "model/ppocrv5_korean_dict.txt",
})
result = engine("scan.png", use_cls=False)
print(result.txts)
```

이 예제는 정방향 이미지를 기준으로 자동 방향 분류를 끕니다. 이 저장소는 **글자 인식 모델**을 제공하며, 검출 모델은 RapidOCR이 별도로 사용합니다. 첫 실행에는 검출 모델을 내려받기 위한 인터넷 연결이 필요할 수 있습니다.

## 인식 성능

모델 공개 당시, 학습에 쓰지 않은 같은 평가 이미지로 기존 `korean_PP-OCRv5_mobile_rec`과 비교한 결과입니다.

단어 평가는 정답 영역으로 잘라낸 이미지를 사용했으며, 글자 사전 밖 문자가 든 단어는 제외했습니다.

- **단어 정확도**: 인식한 문자열 전체가 정답과 일치한 비율입니다. [평가 코드](eval_lists.py)
- **CER(문자 오류율)**: 정답 글자 수 대비 문자 편집 거리입니다. 아래 흘림체 문장 평가는 띄어쓰기를 제외합니다. 낮을수록 좋습니다. [평가 코드](eval_sentences.py)

| 평가 세트 | 기존 PP-OCRv5 | korean-handwriting-rec |
|---|---|---|
| 손글씨 단어 2만 개 | 67.2% | **97.4%** |
| 학습에 없던 필자 15명의 손글씨 6천 단어 | 56.7% | **99.1%** |
| 학습에 없던 정부 서식 5종의 인쇄 글자 6천 단어 | 88.7% | **99.2%** |
| 흘림체 문장 1,300개의 CER, 띄어쓰기 제외 | 30.0% | **7.8%** |

글자 영역을 읽은 결과이며, 문서 전체의 검출 정확도나 표·레이아웃 복원 성능을 뜻하지 않습니다. 흐리거나 겹친 글씨는 잘못 읽을 수 있습니다.

<details>
<summary>세부 평가 결과</summary>

| 평가 세트 | 기존 PP-OCRv5 | korean-handwriting-rec |
|---|---|---|
| 종이에 쓴 서식 | 66.8% | 98.1% |
| 종이에 쓴 자유 글씨 | 73.9% | 96.9% |
| 태블릿에 쓴 서식 | 55.8% | 97.1% |
| 태블릿에 쓴 자유 글씨 | 55.0% | 93.8% |
| 학습에 없던 단어 | 66.7% | 96.3% |
| 기존 모델이 틀린 단어 6,550개 | 0% | 92.9% |
| 정부 서식 중 숫자가 든 단어 | 95.7% | 99.3% |
| 정부 서식에 미리 인쇄된 항목명 | 92.9% | 97.8% |
| 정부 서식에 기재한 값 | 88.4% | 99.3% |

앞의 여섯 행은 손글씨 단어 2만 개의 하위 집합, 뒤의 세 행은 정부 서식 인쇄 글자 6천 단어의 하위 집합입니다.

</details>

### 같은 이미지에서 읽은 결과

| 종류 | 정답 | 기존 PP-OCRv5 | korean-handwriting-rec |
|---|---|---|---|
| 손글씨 | 체납처분유예 | 제납처분예 | 체납처분유예 |
| 손글씨 | 등록면허세 | 등록면해세 | 등록면허세 |
| 손글씨 금액 | 3235원 | 2 | 3235원 |
| 학습에 없던 필자의 손글씨 | 이력서 | 이 령서 | 이력서 |
| 학습에 없던 필자의 손글씨 | 환율 | 화을 | 환율 |
| 서식 인쇄 금액 | 59,470,000 | 59.470.000 | 59,470,000 |
| 서식 인쇄 날짜 | 2017.10.28 | 201710.28 | 2017.10.28 |

## 자주 묻는 질문

### 기존 PP-OCRv5 대신 쓸 수 있나요?

RapidOCR에서는 위 예시처럼 인식 모델과 사전 경로를 지정합니다. 글자 검출 모델은 기존 것을 사용할 수 있습니다. 다른 OCR 도구에 연결할 때는 해당 도구의 입력 전처리와 출력 해석 방식을 확인해야 합니다.

### GPU나 외부 API가 필요한가요?

ONNX Runtime으로 CPU에서 실행할 수 있으며, 인식할 이미지를 외부 OCR API에 보내지 않습니다. 오프라인에서 쓰려면 패키지와 사용할 검출·인식 모델을 미리 준비해야 합니다. 방향 분류를 사용한다면 분류 모델도 필요합니다.

### 인쇄된 글자도 읽나요?

공문서 인쇄 글자를 함께 학습했습니다. 정부 서식 평가 결과와 금액·날짜 인식 예시는 위 표에 있습니다.

### PDF의 표나 문서 구조도 추출하나요?

아니요. 이 모델은 글자 영역의 텍스트를 읽습니다. PDF를 이미지로 바꾸거나 표·레이아웃을 분석하는 기능은 포함하지 않습니다.

## 학습 방법

PaddleOCR의 사전학습 모델을 PyTorch로 옮겨 Mac GPU(MPS)에서 두 단계로 추가 학습했습니다. 글자 사전은 기존 PP-OCRv5 한국어 모델의 것을 유지했습니다.

1. 손글씨 386만 단어를 1 epoch 학습했습니다. 샘플의 20%는 같은 단어를 인쇄체로 렌더링해 섞었습니다.
2. 정부 서식 인쇄 글자 42만 단어와 필기체 30만 단어에 1단계 손글씨 120만 단어를 섞어 이어서 학습했습니다.

[학습](train_rec.py) · [ONNX 내보내기](export_rec.py) · [단어 평가](eval_lists.py) · [문장 평가](eval_sentences.py)

## 데이터 출처와 라이선스

| 데이터 | 용도 |
|---|---|
| [대용량 손글씨 OCR 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=605) | 손글씨 학습·평가 |
| [다양한 형태의 한글 문자 OCR](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=91) | 필기체·공문서 인쇄 글자 학습·평가 |
| [한국어 글자체 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=81) | 흘림체 문장 평가 |

이 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 위 AI Hub 데이터를 활용했습니다. 학습 데이터는 이 저장소에 포함하지 않으며, AI Hub에서 별도로 신청해야 합니다.

모델과 코드는 [Apache-2.0](LICENSE)으로 공개합니다. AI Hub 데이터에는 별도의 이용 조건이 적용됩니다.

기반 모델과 글자 사전은 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), PyTorch 변환은 [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)를 사용했습니다.
