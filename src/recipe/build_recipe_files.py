"""레시피 제출용 파일 3개 생성: 데이터셋 CSV, 컬럼정의 CSV, 데이터 템플릿 예시 CSV.

입력: data/processed/ (최종_순위.csv, 재해_피처.csv)
출력: recipe/
  aidc_siting_dataset.csv            256행 x 21열 (타겟 변수: suitability_score)
  aidc_siting_column_definition.csv  변수 번호 / 타겟 변수 여부 / 데이터 타입 / 데이터 컬럼 명 / 비고
  aidc_siting_template_example.csv   같은 컬럼, 등급별 대표 5행

컬럼 구성 원칙: 식별 3열 -> 피처별(원값, 점수) 5쌍(재해는 4개 세부 재해 점수 포함) -> 타겟 -> 결과·보강 항목.
순위·강점·품질 표시 등 계산 과정의 부산물은 제출용에서 빼고 data/processed/최종_순위.csv에만 둔다.

비고 작성 형식(전 컬럼 공통): "설명. 단위/범위: ... 방향: ... 빈칸: ..." 순서의 짧은 문장.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "recipe"
OUT.mkdir(exist_ok=True)
KEY = "시군구코드"
D_WATER = 4080.0


def read(name):
    return pd.read_csv(PROC / name, encoding="utf-8-sig", dtype={KEY: str})


rk = read("최종_순위.csv")
dz = read("재해_피처.csv")[[KEY, "지진_위험점수_1978_2025", "재해_홍수점수_사용", "산사태_위험점수_2016_2025", "산불_위험점수_2011_2025", "연안_Flag"]]
df = rk.merge(dz, on=KEY, validate="1:1")
assert len(df) == 256 and df[KEY].is_unique

SAFE_NA = df.재해_안전도.notna()  # 연안 고위험(CO Red) 후보제외 10곳은 재해 항목 전체가 빈칸

# (영문 컬럼명, 값, 데이터 타입, 타겟 여부, 비고) -- 이 순서가 CSV 열 순서
SPEC = [
    ("sigungu_code", df[KEY], "String", "X", "시군구 코드(2026년 행정구역 5자리). 형식: 문자열."),
    ("sido_name", df.시도명, "String", "X", "시도 이름."),
    ("sigungu_name", df.시군구명, "String", "X", "시군구 이름(구가 있는 시는 구 단위)."),
    ("power_substation_headroom_mw_2029", df["용량MW_2029"], "Float64", "X",
     "전력 원값. 변전소 접속 가능 여유 용량(2029년 기준, 단일 변전소 최댓값). 단위: MW. 방향: 클수록 유리. 빈칸: 전력 자료 없음."),
    ("power_score", df.점수_전력, "Float64", "X",
     "전력 점수. 전력 원값의 min-max 정규화(평가 대상 245곳 기준). 범위: 0~1. 방향: 클수록 유리. 빈칸: 후보제외 또는 전력 자료 없음."),
    ("renewable_generation_mwh_2024", df.신재생_발전량_MWh, "Float64", "X",
     "신재생 원값. 2024년 재생에너지 발전량. 단위: MWh. 방향: 클수록 유리."),
    ("renewable_score", df.점수_신재생, "Float64", "X",
     "신재생 점수. log(1+원값)의 min-max 정규화(평가 대상 245곳 기준). 범위: 0~1. 방향: 클수록 유리. 빈칸: 후보제외."),
    ("water_supply_headroom_m3_per_day", df.수자원_H, "Float64", "X",
     "수자원 원값. 최대급수일 기준 물 공급 여유(자체 정수장 + 광역 정수장 + 이웃 공급). 단위: m³/일. 방향: 클수록 유리. 기준: 4,080 미만이면 용수선결 등급."),
    ("water_score", df.점수_수자원, "Float64", "X",
     "수자원 점수. log(1+원값)의 min-max 정규화(평가 대상 245곳 기준). 범위: 0~1. 방향: 클수록 유리. 빈칸: 후보제외."),
    ("disaster_earthquake_score", df.지진_위험점수_1978_2025.where(SAFE_NA), "Float64", "X",
     "지진 위험 점수(1978~2025 지진 기록 기반). 범위: 0~1. 방향: 클수록 위험. 안전도 가중치: 0.19. 빈칸: 후보제외(연안 고위험)."),
    ("disaster_flood_score", df.재해_홍수점수_사용.where(SAFE_NA), "Float64", "X",
     "홍수 위험 점수(100년 빈도 하천 침수 예상 면적 기반). 범위: 0~1. 방향: 클수록 위험. 안전도 가중치: 0.17. 빈칸: 후보제외(연안 고위험)."),
    ("disaster_landslide_score", df.산사태_위험점수_2016_2025.where(SAFE_NA), "Float64", "X",
     "산사태 위험 점수(2016~2025 산사태 기록 기반). 범위: 0~1. 방향: 클수록 위험. 안전도 가중치: 0.13. 빈칸: 후보제외(연안 고위험)."),
    ("disaster_wildfire_score", df.산불_위험점수_2011_2025.where(SAFE_NA), "Float64", "X",
     "산불 위험 점수(2011~2025 산불 기록 기반). 범위: 0~1. 방향: 클수록 위험. 안전도 가중치: 0.08. 빈칸: 후보제외(연안 고위험)."),
    ("disaster_safety_index", df.재해_안전도, "Float64", "X",
     "재해 원값. 1 − 위험지수(위 4개 위험 점수의 가중합, 연안 CO Yellow는 ×1.2). 범위: 0~1. 방향: 클수록 안전. 빈칸: 후보제외(연안 고위험)."),
    ("disaster_score", df.점수_재해, "Float64", "X",
     "재해 점수. 안전도의 min-max 정규화(평가 대상 245곳 기준). 범위: 0~1. 방향: 클수록 유리. 빈칸: 후보제외."),
    ("cooling_free_cooling_hour_ratio", df["냉각_WSE가능비율"], "Float64", "X",
     "냉각 원값. 습구온도 12.8℃ 이하인 시간의 비율(외기 냉각 가능 시간, 2024~2025 평균). 범위: 0~1. 방향: 클수록 유리."),
    ("cooling_score", df.점수_냉각, "Float64", "X",
     "냉각 점수. 냉각 원값의 min-max 정규화(평가 대상 245곳 기준). 범위: 0~1. 방향: 클수록 유리. 빈칸: 후보제외."),
    ("suitability_score", df.수용점수, "Float64", "O",
     "타겟. 5개 점수의 가중합(전력 41.75%, 신재생 19.22%, 수자원 16.39%, 재해 12.97%, 냉각 9.68%). 범위: 0~1. 방향: 클수록 유리. 빈칸: 후보제외(전력이 빈 3곳은 나머지 점수로 계산)."),
    ("suitability_grade", df.등급, "String", "X",
     "수용 등급. 값: A·B·C(용수 통과 지역의 상위 20%·20~50%·하위 50%), 용수선결(물 공급 여유 4,080 m³/일 미만), 판정보류(전력 자료 없음), 제외(후보제외)."),
    ("reinforcement_needs", df.보강항목, "String", "X",
     "보강이 필요한 항목. 값: 전력·용수·신재생·재해·냉각 중 해당 항목(쉼표 구분). 기준: 전력 여유 0MW 또는 하위 25%, 용수 공급 여유 4,080 미만, 신재생·재해·냉각 하위 25%. 빈칸: 보강 항목 없음."),
    ("excluded_reason", df.제외사유.replace("", np.nan), "String", "X",
     "후보제외 사유. 값: 연안 고위험(CO Red) 10곳, 전력 분석 제외(울릉군) 1곳. 빈칸: 제외 아님."),
]

names = [s[0] for s in SPEC]
assert len(set(names)) == len(names)
data = pd.DataFrame({s[0]: s[1].values for s in SPEC})

for c in data.columns:
    if c.endswith("_score") or c in ("disaster_safety_index", "suitability_score"):
        data[c] = data[c].round(4)
    elif c == "cooling_free_cooling_hour_ratio":
        data[c] = data[c].round(4)
    elif c.endswith("_mwh_2024"):
        data[c] = data[c].round(1)
    elif c.endswith("_m3_per_day"):
        data[c] = data[c].round(0)

# ---------- 검수 ----------
g = data.suitability_grade
assert data.sigungu_code.str.fullmatch(r"\d{5}").all()
sc = [c for c in data.columns if c.endswith("_score")]
assert (data[sc].stack().dropna().between(0, 1)).all()
assert data.suitability_score.isna().equals(g == "제외"), "타겟 결측은 제외 11곳과 같아야 함"
assert (g == "제외").sum() == 11 and g.isin(["A", "B", "C"]).sum() == 201
assert (data.loc[g == "용수선결", "water_supply_headroom_m3_per_day"] < D_WATER).all()
assert (data.loc[g.isin(["A", "B", "C"]), "water_supply_headroom_m3_per_day"] >= D_WATER).all()
assert data.excluded_reason.notna().equals(g == "제외")

# 재해 세부 점수 4개가 안전도를 그대로 재현하는지 확인(안전도 = 1 - 가중합, 연안 CO Yellow는 x1.2)
m = data.disaster_safety_index.notna()
yellow = df.연안_Flag.eq("CO Yellow").map({True: 1.2, False: 1.0})
risk = (0.19 * data.disaster_earthquake_score + 0.17 * data.disaster_flood_score
        + 0.13 * data.disaster_landslide_score + 0.08 * data.disaster_wildfire_score) / 0.57 * yellow
assert (1 - risk[m] - data.disaster_safety_index[m]).abs().max() < 2e-4, "세부 점수로 안전도가 재현되지 않음"
sub = [c for c in data.columns if c.startswith("disaster_") and c.endswith("_score") and c != "disaster_score"]
assert data.loc[m, sub].notna().all().all()

data.to_csv(OUT / "aidc_siting_dataset.csv", index=False, encoding="utf-8-sig")

# ---------- 컬럼 정의 ----------
rows = [{"변수 번호": i, "타겟 변수 여부": tgt, "데이터 타입": typ, "데이터 컬럼 명": name, "비고": note}
        for i, (name, _, typ, tgt, note) in enumerate(SPEC, 1)]
pd.DataFrame(rows).to_csv(OUT / "aidc_siting_column_definition.csv", index=False, encoding="utf-8-sig")

# ---------- 템플릿 예시(등급별 대표 5행) ----------
pick = []
for grade in ("A", "B", "C"):
    cand = data[(data.suitability_grade == grade) & data.reinforcement_needs.notna()]
    pick.append(cand.iloc[len(cand) // 2].name)
pick.append(data[data.suitability_grade == "용수선결"].sort_values("suitability_score", ascending=False).iloc[0].name)
pick.append(data[data.suitability_grade == "제외"].sort_values("sigungu_code").iloc[0].name)
tpl = data.loc[pick]
tpl.to_csv(OUT / "aidc_siting_template_example.csv", index=False, encoding="utf-8-sig")

print(f"데이터셋 {data.shape}, 컬럼정의 {len(rows)}행, 템플릿 {tpl.shape}")
print("등급 분포:", g.value_counts().to_dict())
print("타겟 결측:", int(data.suitability_score.isna().sum()), "| 타겟 범위:", data.suitability_score.min(), data.suitability_score.max())
