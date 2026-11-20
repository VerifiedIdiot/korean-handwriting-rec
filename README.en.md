# korean-handwriting-rec · Korean handwriting OCR

**PP-OCRv5 for handwritten Korean.**

[한국어](README.md) | English · [Download the model](https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec)

korean-handwriting-rec adapts PP-OCRv5 to handwritten Korean through additional training on AI Hub handwriting and printed government forms. The ONNX model is about 15 MB and runs on a CPU.

- **Handwritten symbols: 39.7% → 97.3%**, for strings such as `₩8,470,000`, `143.1m²`, `2026.7.1~2028.6.30`, `12+30=42` and `〃`.
- **Commas and periods in handwritten amounts and dates: 56.8% → 96.9%**, for strings such as `7,278.37` and `(679,000)`.
- **Handwritten words: 67.2% → 95.9%** on 20,000 words outside the training set.
- **Cursive sentence error rate: 30.0% → 6.3%**

[Evaluation results](#recognition-results)

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

These results compare the model with `korean_PP-OCRv5_mobile_rec`. Both models used the same evaluation images, which were outside the training set.

Word evaluation used crops from annotated text regions. It excluded words with characters outside the dictionary.

- **Word accuracy** measures exact matches between the complete output string and the reference string. [Evaluation code](eval_lists.py)
- **Character error rate (CER)** divides character edit distance by the reference character count. The cursive sentence evaluation excludes spaces. Lower is better. [Evaluation code](eval_sentences.py)

| Evaluation set | Original PP-OCRv5 | korean-handwriting-rec |
|---|---|---|
| Handwritten words, 20,000 samples | 67.2% | **95.9%** |
| Handwriting from 15 unseen writers, 6,000 words | 56.7% | **99.2%** |
| Printed text from 5 unseen government forms, 6,000 words | 88.7% | **99.2%** |
| Handwritten amounts, dates and phone numbers, 3,000 lines (commas, periods, hyphens, brackets) | 56.8% | **96.9%** |
| Handwritten amounts, rates, areas and periods with symbols, 3,000 lines (₩ 〃 % + = # ² ~) | 39.7% | **97.3%** |
| Cursive sentences, 1,300 samples, CER without spaces | 30.0% | **6.3%** |

<details>
<summary>Detailed results</summary>

| Evaluation subset | Original PP-OCRv5 | korean-handwriting-rec |
|---|---|---|
| Handwritten forms on paper | 66.8% | 97.7% |
| Freehand text on paper | 73.9% | 93.9% |
| Handwritten forms on tablets | 55.8% | 95.8% |
| Freehand text on tablets | 55.0% | 92.2% |
| Words absent from the training set | 66.6% | 95.5% |
| 6,550 words that the original model read incorrectly | 0% | 89.4% |
| Words with digits in government forms | 95.7% | 99.3% |
| Preprinted field labels in government forms | 92.9% | 98.0% |
| Field values in government forms | 88.4% | 99.3% |
| Slashes in handwritten dates and page numbers | 61.5% | 98.2% |

The first six rows are subsets of the 20,000 handwritten words, the next three are subsets of the 6,000 printed form words, and the last is a subset of the 3,000 handwritten number lines.

</details>

### Outputs from the same images

| Text type | Reference | Original PP-OCRv5 | korean-handwriting-rec |
|---|---|---|---|
| Handwritten word | 체납처분유예 | 제납처분예 | 체납처분유예 |
| Handwritten word | 등록면허세 | 등록면해세 | 등록면허세 |
| Handwritten word | 환율 | 환 | 환율 |
| Handwritten amount | 3235원 | 3275원 | 3235원 |
| Handwritten decimal | 7,278.37 | 1298.319 | 7,278.37 |
| Handwritten amount in brackets | (679,000) | (619,00) | (679,000) |
| Handwritten amount with ₩ | ₩8,470,000 | 8,410,000 | ₩8,470,000 |
| Handwritten area | 143.1m² | 143./m | 143.1m² |
| Handwritten period | 2026~2027 | 2026~202n | 2026~2027 |
| Handwritten sum | 2044+9207=11251 | 2044+9207=1/251 | 2044+9207=11251 |

## Questions

### Can this model replace PP-OCRv5 in an existing pipeline?

In RapidOCR, set the recognition model and dictionary paths as shown above. You can retain your text detection model. For other OCR tools, check their image preprocessing and output decoding requirements.

### Does inference require a GPU or an external API?

ONNX Runtime can run the model on a CPU without an external OCR API. Before offline use, prepare the packages and your detection and recognition models. If you enable orientation classification, prepare that model too.

### Does it recognize printed text?

The model also uses printed government forms for training and reads 99.2% of printed words from 5 unseen government forms.

### Does it read handwritten amounts and dates?

Yes. Handwritten digits mixed with commas, periods, hyphens, slashes and brackets, plus the won sign (₩), ditto mark (〃), percent, plus, equals, number sign, superscript (m²) and tilde have their own training data. The model reads 96.9% of 3,000 handwritten amount, date and phone number lines, and 97.3% of 3,000 lines with handwritten symbols.

### Is the character dictionary the same as PP-OCRv5?

It adds `₩` and `〃`, two symbols that people often write by hand on forms, to the end of the 11,945-character PP-OCRv5 Korean dictionary (11,947 characters). Swap the model and dictionary files together.

### Does it extract PDF tables or document structure?

No. The model recognizes text within image regions. It does not convert PDFs to images, extract tables, or analyze document layouts.

## Model training

The model uses pretrained PaddleOCR weights with a PyTorch conversion and four training stages. The dictionary is the PP-OCRv5 Korean dictionary with `₩` and `〃` added.

1. **Handwriting** (Mac GPU): 3.86 million handwritten words for one epoch, with 20% of the samples rendered as printed text.
2. **Printed forms** (Mac GPU): 420,000 printed form words, 300,000 handwritten words, and 1.2 million handwritten words from the first stage.
3. **Digits, symbols and more writers** (NVIDIA DGX Spark): 400,000 amount, date and phone number lines built from each writer's own handwritten digits with commas, periods, slashes, brackets and the won sign, plus 580,000 words written by elementary and middle school students, mixed with the earlier data for 1.9 million samples.
4. **Handwritten symbols** (NVIDIA DGX Spark): the output layer grows for `₩` and `〃`, and 150,000 lines of rates, areas, periods, sums, numbers and ditto marks written with handwritten digits are mixed with the earlier data for 500,000 samples.

[Training](train_rec.py) · [Digit and symbol lines](make_synth_digits.py) · [Training mix](build_mix.py) · [ONNX export](export_rec.py) · [Word evaluation](eval_lists.py) · [Sentence evaluation](eval_sentences.py)

## Data sources and license

| AI Hub dataset | Use |
|---|---|
| [대용량 손글씨 OCR 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=605) | Handwritten word training and evaluation |
| [다양한 형태의 한글 문자 OCR](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=91) | Handwritten and printed form text training and evaluation |
| [OCR 데이터(교육)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71300) | Student handwriting training |
| [한국어 글자체 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=81) | Cursive sentence evaluation |

The datasets come from The Open AI Dataset Project (AI-Hub, South Korea), with support from MSIT and NIA. This repository does not distribute the datasets. Request access through AI Hub.

The model and code use the [Apache-2.0 license](LICENSE). AI Hub datasets have separate terms of use.

The base model and dictionary come from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). The PyTorch conversion uses [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch).
