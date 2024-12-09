import json
from dataclasses import dataclass
from enum import Enum


class T_category(Enum):
    """
    T分類 原発腫瘍サイズ
    """

    T0 = "T0"  # "原発腫瘍を認めない"
    Tis = "Tis"  # "上皮内癌（carcinoma in situ）：肺野型の場合は，充実成分径0cmかつ病変全体径≦3cm"

    # T1x
    # 腫瘍の充実成分径<=3cm，肺または臓側胸膜に覆われている
    # 葉気管支より中枢への浸潤が気管支鏡上認められない（すなわち主気管支に及んでいない）
    T1mi = "T1mi"  # 微少浸潤性腺癌：部分充実型を示し，充実成分径≦0.5cmかつ病変全体径≦3cm
    T1a = "T1a"  # 充実成分径≦1cmでかつTis・T1miには相当しない
    T1b = "T1b"  # 充実成分径＞1cmでかつ≦2cm
    T1c = "T1c"  # 充実成分径＞2cmでかつ≦3cm

    # T2x
    # 充実成分径>3cmでかつ<=5cm，または充実成分径<=3cmでも以下のいずれかであるもの
    # - 主気管支に及ぶが気管分岐部には及ばない
    # - 臓側胸膜に浸潤
    # - 肺門まで連続する部分的または一側全体の無気肺か閉塞性肺炎がある
    T2a = "T2a"  # 充実成分径＞3cmでかつ≦4cm"
    T2b = "T2b"  # 充実成分径＞4cmでかつ≦5cm"
    T3 = "T3"  # 充実成分径＞5cmでかつ≦7cm，または充実成分径≦5cmでも以下のいずれかであるもの\n壁側胸膜，胸壁（superior sulcus tumorを含む），横隔神経，心膜のいずれかに直接浸潤\n同一葉内の不連続な副腫瘍結節"
    T4 = "T4"  # 充実成分径＞7cm，または大きさを問わず横隔膜，縦隔，心臓，大血管，気管，反回神経，食道，椎体，気管分岐部への浸潤，あるいは同側の異なった肺葉内の副腫瘍結節"


class N_category(Enum):
    """
    N分類 所属リンパ節
    """

    N0 = "N0"  # 所属リンパ節転移なし
    N1 = "N1"  # 同側の気管支周囲かつ/または同側肺門，肺内リンパ節への転移で原発腫瘍の直接浸潤を含める
    N2 = "N2"  # 同側縦隔かつ/または気管分岐下リンパ節への転移
    N3 = "N3"  # 対側縦隔，対側肺門，同側あるいは対側の前斜角筋，鎖骨上窩リンパ節への転移


class M_category(Enum):
    """
    遠隔転移のTNM分類
    """

    M0 = "M0"  # 遠隔転移なし
    # M1x: 遠隔転移がある
    M1a = "M1a"  # 対側肺内の副腫瘍結節，胸膜または心膜の結節，悪性胸水（同側・対側），悪性心嚢水
    M1b = "M1b"  # 肺以外の一臓器への単発遠隔転移がある
    M1c = "M1c"  # 肺以外の一臓器または多臓器への多発遠隔転移がある


@dataclass(frozen=True)
class MainTaskResult:
    record_id: str
    t: T_category
    n: N_category
    m: M_category

    def json(self) -> str:
        return json.dumps({"record_id": self.record_id, "t": self.t.name, "n": self.n.name, "m": self.m.name})
