# korean-handwriting-rec · Korean handwriting OCR

**PP-OCRv5 for handwritten Korean.**

[한국어](README.md) | English · [Download the model](https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec)

korean-handwriting-rec adapts PP-OCRv5 to handwritten Korean through additional training on AI Hub handwriting and printed government forms. The ONNX model is about 15 MB and runs on a CPU.

**Handwritten word accuracy: 67.2% → 97.4%.** Both models used the same 20,000 words, which were outside the training set. [Evaluation results](#recognition-results)

In RapidOCR, set the recognition model and character dictionary paths to use this model.

## Quick start

Clone the repository. Install RapidOCR 3.x with ONNX Runtime.

```bash
git clone https://github.com/VerifiedIdiot/korean-handwriting-rec.git
cd korean-handwriting-rec
python -m pip install "rapidocr>=3,<4" onnxruntime
```

Run this example from the repository root. Replace `scan.png` with the path to your image.

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Rec.model_path": "model/rec_korean_hw.onnx",
    "Rec.rec_keys_path": "model/ppocrv5_korean_dict.txt",
})
result = engine("scan.png", use_cls=False)
print(result.txts)
```

This example expects upright text and disables orientation classification. This repository provides the **text recognition model**. RapidOCR uses a separate text detection model, which can require a download on its first run.

## Recognition results

These results come from the model's original evaluation against `korean_PP-OCRv5_mobile_rec`. Both models used the same evaluation images, which were outside the training set.

Word evaluation used crops from annotated text regions. It excluded words with characters outside the dictionary.

- **Word accuracy** measures exact matches between the complete output string and the reference string. [Evaluation code](eval_lists.py)
- **Character error rate (CER)** divides character edit distance by the reference character count. The cursive sentence evaluation excludes spaces. Lower is better. [Evaluation code](eval_sentences.py)

| Evaluation set | Original PP-OCRv5 | korean-handwriting-rec |
|---|---|---|
| Handwritten words, 20,000 samples | 67.2% | **97.4%** |
| Handwriting from 15 unseen writers, 6,000 words | 56.7% | **99.1%** |
| Printed text from 5 unseen government forms, 6,000 words | 88.7% | **99.2%** |
| Cursive sentences, 1,300 samples, CER without spaces | 30.0% | **7.8%** |

These results measure recognition within text regions, not full-page detection, table extraction, or layout analysis. Blurred or overlapping text can produce recognition errors.

<details>
<summary>Detailed results</summary>

| Evaluation subset | Original PP-OCRv5 | korean-handwriting-rec |
|---|---|---|
| Handwritten forms on paper | 66.8% | 98.1% |
| Freehand text on paper | 73.9% | 96.9% |
| Handwritten forms on tablets | 55.8% | 97.1% |
| Freehand text on tablets | 55.0% | 93.8% |
| Words absent from the training set | 66.7% | 96.3% |
| 6,550 words that the original model read incorrectly | 0% | 92.9% |
| Words with digits in government forms | 95.7% | 99.3% |
| Preprinted field labels in government forms | 92.9% | 97.8% |
| Field values in government forms | 88.4% | 99.3% |

The first six rows are subsets of the 20,000 handwritten words. The last three rows are subsets of the 6,000 printed form words.

</details>

### Outputs from the same images

| Text type | Reference | Original PP-OCRv5 | korean-handwriting-rec |
|---|---|---|---|
| Handwritten word | 체납처분유예 | 제납처분예 | 체납처분유예 |
| Handwritten word | 등록면허세 | 등록면해세 | 등록면허세 |
| Handwritten amount | 3235원 | 2 | 3235원 |
| Handwriting from an unseen writer | 이력서 | 이 령서 | 이력서 |
| Handwriting from an unseen writer | 환율 | 화을 | 환율 |
| Printed amount | 59,470,000 | 59.470.000 | 59,470,000 |
| Printed date | 2017.10.28 | 201710.28 | 2017.10.28 |

## Questions

### Can this model replace PP-OCRv5 in an existing pipeline?

In RapidOCR, set the recognition model and dictionary paths as shown above. You can retain your text detection model. For other OCR tools, check their image preprocessing and output decoding requirements.

### Does inference require a GPU or an external API?

ONNX Runtime can run the model on a CPU without an external OCR API. Before offline use, prepare the packages and your detection and recognition models. If you enable orientation classification, prepare that model too.

### Does it recognize printed text?

The model also uses printed government forms for training. The tables above include printed text results and examples with amounts and dates.

### Does it extract PDF tables or document structure?

No. The model recognizes text within image regions. It does not convert PDFs to images, extract tables, or analyze document layouts.

## Model training

The model uses pretrained PaddleOCR weights with a PyTorch conversion. Training used a Mac GPU through MPS in two stages. The PP-OCRv5 Korean character dictionary remains unchanged.

1. The first stage used 3.86 million handwritten words for one epoch. It rendered 20% of the samples as printed text.
2. The second stage mixed 420,000 printed form words, 300,000 handwritten words, and 1.2 million handwritten words from the first stage.

[Training](train_rec.py) · [ONNX export](export_rec.py) · [Word evaluation](eval_lists.py) · [Sentence evaluation](eval_sentences.py)

## Data sources and license

| AI Hub dataset | Use |
|---|---|
| [대용량 손글씨 OCR 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=605) | Handwritten word training and evaluation |
| [다양한 형태의 한글 문자 OCR](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=91) | Handwritten and printed form text training and evaluation |
| [한국어 글자체 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=81) | Cursive sentence evaluation |

The datasets come from The Open AI Dataset Project (AI-Hub, South Korea), with support from MSIT and NIA. This repository does not distribute the datasets. Request access through AI Hub.

The model and code use the [Apache-2.0 license](LICENSE). AI Hub datasets have separate terms of use.

The base model and dictionary come from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). The PyTorch conversion uses [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch).
