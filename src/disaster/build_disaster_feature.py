"""재해 피처 최종: 세원 님 07_재해_점수화_최종.csv에서 홍수지도 미확보 행을 처리해 256행 파일을 만든다.
- 기본 4재해지수 = (0.19*지진 + 0.17*홍수 + 0.13*산사태 + 0.08*산불) / 0.57, 최종 = 기본 x 연안 Yellow 계수(1.2 / 1.0)
- 연안 CO Red 10행은 후보제외(위험지수 비움)
- 홍수지도 미확보 행의 홍수 점수 처리
    하천 없음(국가·지방하천 지정 없음): 0  -> 옹진군, 영종구, 미추홀구, 제물포구, 목포시(국가하천 포함 여부 미확인)
    하천은 있으나 지도 없음: 홍수 점수 25분위 -> 신안군(지방하천 고란천·방월천)
"""
import sys

import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "disaster" / "07_재해_점수화_최종.csv"
OUT = ROOT / "data" / "processed"
W = dict(E=0.19, F=0.17, L=0.13, W=0.08)
NO_RIVER = {"12110": "목포시", "28125": "제물포구", "28155": "영종구", "28177": "미추홀구", "28720": "옹진군"}
HAS_RIVER_NO_MAP = {"12870": "신안군"}

d = pd.read_csv(SRC, encoding="utf-8-sig", dtype={"시군구코드": str})
cols = {"E": "지진_위험점수_1978_2025", "F": "홍수_위험점수_100년빈도", "L": "산사태_위험점수_2016_2025",
        "W": "산불_위험점수_2011_2025", "base": "기본_4재해위험지수", "final": "최종_재해위험지수", "y": "연안_Yellow_가중계수"}
for c in cols.values():
    d[c] = pd.to_numeric(d[c], errors="coerce")

def index(flood, y=None):
    x = (W["E"] * d[cols["E"]] + W["F"] * flood + W["L"] * d[cols["L"]] + W["W"] * d[cols["W"]]) / 0.57
    return x * (d.연안_Flag.eq("CO Yellow").map({True: 1.2, False: 1.0}) if y is None else y)

# 0) 원본 재현 검증: 홍수가 있는 행은 파일의 최종지수와 일치해야 한다
ok = d[cols["final"]].notna()
assert (index(d[cols["F"]])[ok] - d.loc[ok, cols["final"]]).abs().max() < 1e-5

# 1) 홍수 처리
q25 = d[cols["F"]].quantile(0.25)
d["재해_홍수점수_사용"] = d[cols["F"]]
d["재해_홍수_처리"] = np.where(d[cols["F"]].notna(), "원자료", "")
miss = d.홍수_자료상태.eq("지도미확보")
for code in NO_RIVER:
    i = d.index[d.시군구코드 == code][0]
    assert miss[i], code
    d.loc[i, ["재해_홍수점수_사용", "재해_홍수_처리"]] = [0.0, "하천없음_0"]
for code in HAS_RIVER_NO_MAP:
    i = d.index[d.시군구코드 == code][0]
    assert miss[i], code
    d.loc[i, ["재해_홍수점수_사용", "재해_홍수_처리"]] = [q25, "하천있음_지도없음_25분위대체"]
# 위 6곳 밖의 지도미확보 행(부산 중구·서구·영도구)은 CO Red 후보제외라 위험지수를 비운다
left = d[miss & d.재해_홍수_처리.eq("")]
assert left.연안_Flag.eq("CO Red").all(), left[["시군구", "연안_Flag"]]
d.loc[left.index, "재해_홍수_처리"] = "미확보_후보제외"

# 2) 위험지수 / 안전도
red = d.연안_Flag.eq("CO Red")
risk = index(pd.to_numeric(d.재해_홍수점수_사용))
d["재해_후보제외"] = red
d["재해_위험지수"] = risk.where(~red & d.재해_홍수점수_사용.notna())
d["재해_안전도"] = 1 - d.재해_위험지수
d["재해_홍수대체"] = d.재해_홍수_처리.isin(["하천없음_0", "하천있음_지도없음_25분위대체"])

# 3) 민감도: 대체 행에 홍수=0 / 25분위 / 중앙값 적용 시 위험지수
med = d[cols["F"]].median()
for lab, v in [("홍수0", 0.0), ("홍수25분위", q25), ("홍수중앙값", med)]:
    s = index(pd.Series(v, index=d.index))
    d[f"재해_위험지수_{lab}"] = s.where(d.재해_홍수대체 & ~red)

out_cols = ["시군구코드", "시도", "시군구", cols["E"], cols["F"], cols["L"], cols["W"], "연안_Flag", "재해_홍수점수_사용",
            "재해_홍수_처리", "재해_홍수대체", "재해_후보제외", "재해_위험지수", "재해_안전도",
            "재해_위험지수_홍수0", "재해_위험지수_홍수25분위", "재해_위험지수_홍수중앙값", "최종_재해위험지수_상태"]
d[out_cols].to_csv(OUT / "재해_피처.csv", index=False, encoding="utf-8-sig")

print("행", len(d), "| 위험지수 있음", d.재해_위험지수.notna().sum(), "| 후보제외", int(red.sum()), "| 홍수대체", int(d.재해_홍수대체.sum()))
print("홍수 25분위", round(q25, 3), "중앙값", round(med, 3))
ref = d.재해_위험지수.dropna()
t = d[d.재해_홍수대체 & ~red][["시군구", "재해_홍수_처리", "재해_위험지수", "재해_위험지수_홍수0", "재해_위험지수_홍수25분위", "재해_위험지수_홍수중앙값"]].copy()
t["위험백분위"] = t.재해_위험지수.apply(lambda v: round((ref < v).mean() * 100))
print(t.round(3).to_string(index=False))
print(d.재해_홍수_처리.value_counts().to_dict())
print(d.재해_안전도.describe().round(3).to_dict())
