import json
import textwrap
from string import Template

from pydantic import BaseModel

from radnlp2024.genai import GenerativeModel, text_completion
from radnlp2024.models import M_category, N_category, T_category

DEFAULT_PROMPT_TEAMPLATE = """
## 指示
肺癌の読影レポート(画像診断報告書)から，肺癌のステージ(進行度)を判定してください。
肺癌のステージはT因子・N因子・M因子で判定されます。3つの因子の説明を元に与える読影レポートについて3つの因子を判定してください。

## 分類仕様

### T因子(原発腫瘍サイズ)
- TX: 原発腫瘍の存在が判定できない，あるいは喀痰または気管支洗浄液細胞診でのみ陽性で画像診断や気管支鏡では観察できない
- T0: 原発腫瘍を認めない
- Tis: 上皮内癌（carcinoma in situ）肺野型の場合は、充実成分径0cmかつ病変全体径≦3cm
- T1: 腫瘍の充実成分径<=3cm, 肺または臓側胸膜に覆われている，葉気管支より中枢への浸潤が気管支鏡上認められない（すなわち主気管支に及んでいない）
  - T1mi: 微少浸潤性腺癌:部分充実型を示し，充実成分径<=0.5cmかつ病変全体径<=3cm
  - T1a: 充実成分径<=1cmでかつTis・T1miには相当しない
  - T1b: 充実成分径>1cmでかつ<=2cm
  - T1c: 充実成分径>2cmでかつ<=3cm
- T2: 充実成分径>3cmでかつ<=5cm、または充実成分径<=3cmでも以下のいずれかであるもの。主気管支に及ぶが気管分岐部には及ばない。臓側胸膜に浸潤。肺門まで連続する部分的または一側全体の無気肺か閉塞性肺炎がある
  - T2a: 充実成分径>3cmでかつ<=4cm
  - T2b: 充実成分径>4cmでかつ<=5cm
- T3: 充実成分径>5cmでかつ≦7cm，または充実成分径≦5cmでも以下のいずれかであるもの。壁側胸膜、胸壁（superior sulcus tumorを含む）、横隔神経、心膜のいずれかに直接浸潤 同一葉内の不連続な副腫瘍結節
- T4: 充実成分径>7cm、または大きさを問わず横隔膜、縦隔、心臓、大血管、気管、反回神経、食道、椎体、気管分岐部への浸潤、あるいは同側の異なった肺葉内の副腫瘍結節

### N因子(所属リンパ節)
- N0: 所属リンパ節転移なし
- N1: 同側の気管支周囲かつ/または同側肺門，肺内リンパ節への転移で原発腫瘍の直接浸潤を含める
- N2: 同側縦隔かつ/または気管分岐下リンパ節への転移
- N3: 対側縦隔，対側肺門，同側あるいは対側の前斜角筋，鎖骨上窩リンパ節への転移

### M因子(遠隔転移)
- M0: 遠隔転移なし
- M1a: 遠隔転移あり、対側肺内の副腫瘍結節、胸膜または心膜の結節、悪性胸水（同側・対側）、悪性心嚢水
- M1b: 遠隔転移あり、肺以外の一臓器への単発遠隔転移がある
- M1c: 遠隔転移あり、肺以外の一臓器または多臓器への多発遠隔転移がある

## 出力フォーマット

JSONでT因子をt, N因子をn, M因子をmとして出力してください。

## 入出力例

${few_shots}

## 処理対象レポート

input:
${test_input}

output:
"""


class ClassificationResult(BaseModel):
    t: T_category
    n: N_category
    m: M_category


def classify_lung_cancer_staging(
    radiology_report: str,
    few_shots: str,
    prompt_template: str = DEFAULT_PROMPT_TEAMPLATE,
    generative_model: GenerativeModel = GenerativeModel.Gemini15Flash001,
) -> tuple[ClassificationResult, str]:
    composed_prompt = Template(prompt_template).substitute(
        few_shots=few_shots,
        test_input=radiology_report,
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "dialogue_response",
            "schema": {
                "type": "object",
                "properties": {
                    "t": {"type": "string", "enum": [e.name for e in T_category]},
                    "n": {"type": "string", "enum": [e.name for e in N_category]},
                    "m": {"type": "string", "enum": [e.name for e in M_category]},
                },
                "required": ["t", "n", "m"],
            },
        },
        "strict": True,
    }

    success, response, finish_reason = text_completion(
        generative_model=generative_model,
        messages=[("user", composed_prompt)],
        temperature=0.0,
        response_format=response_format,
    )
    parsed_response = json.loads(response)

    return ClassificationResult(**parsed_response), composed_prompt


if __name__ == "__main__":
    # From sample dataset #1 (Correct label is T3, N1, M1b)
    radiology_report = textwrap.dedent("""
    肺野背景に気腫性変化を認めます。
    左肺下葉に辺縁不整な腫瘤を認めます。長径 35mm 程度です。既知の肺癌と考えます。
    この腫瘤以外に左肺下葉には径 8mm の結節を認めます。転移を疑います。
    その他に肺転移を疑う病変は指摘できません。
    左肺門部に病的腫大リンパ節を認め、転移を疑います。
    縦隔や鎖骨上窩のリンパ節に病的腫大は認めません。
    胸水貯留はありません。
    肝には胆嚢床に境界不明瞭な乏血性腫瘤を 2 ヶ所認めます。転移を疑います。
    この腫瘤と胆嚢は接しており、胆嚢にはびまん性の壁肥厚を認めます。胆嚢への浸潤の可能性が考えられます。
    副腎に転移を疑う病変はありません。
    両腎嚢胞を認めます。
    その他の腹部臓器に有意な異常所見は指摘できません。
    腹部リンパ節に病的腫大はありません。
    腹水貯留はありません。
    強く骨転移を示唆する骨破壊や骨硬化像は認めません。
    """).strip()

    # From sample dataset #2, #3
    few_shots = textwrap.dedent("""
    input:
    肺野背景に慢性肺気腫の所見を疑います。
    右肺尖部に胸膜に広範囲に接する spicula を伴う⻑径 50mm の腫瘤を認めます。肺癌を疑います。臓側胸膜浸潤を疑います。肺野には明らかな副腫瘍結節を認めません。
    縦隔・肺門部リンパ節の有意な腫大、その他の縦隔器質病変は認めません。
    胸水は認めません。
    撮像範囲の上腹部臓器に明らかな異常は認めません。

    output:
    {"t": "T1b", "n": "N0", "m": "M0"}

    input:
    右肺尖部に最大径 46mm  の分葉状腫瘤があります。
    周囲にすりガラス影を伴っています。
    肺癌と考えます。
    胸水貯留は認めません。
    右肺門、同側、対側縦隔リンパ節腫大しています。
    リンパ節転移と考えます。
    胸水貯留は認めません。
    左副腎は腫大しています。
    副腎転移と考えます。

    output:
    {"t": "T2b", "n": "N1", "m": "M0"}
    """).strip()
    ret, prompt = classify_lung_cancer_staging(radiology_report, few_shots)
    print(ret)
