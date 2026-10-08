"""레시피 제출용 파일 3개 생성: 데이터셋 CSV, 컬럼정의 CSV, 데이터 템플릿 예시 CSV.

입력: data/processed/ (최종_순위.csv, 수자원_냉각_통합.csv, 재해_피처.csv)
출력: recipe/
  aidc_siting_dataset.csv            256행 x 21열 (타겟 변수: suitability_score)
  aidc_siting_column_definition.csv  변수 번호 / 타겟 변수 여부 / 데이터 타입 / 데이터 컬럼 명 / 비고
  aidc_siting_template_example.csv   같은 컬럼, 등급별 대표 5행

컬럼 구성 원칙: 식별 3열 -> 피처별(원값, 점수) 5쌍(재해는 4개 세부 재해 점수 포함) -> 타겟 -> 결과·보강 항목.
순위·강점·품질 표시 등 계산 과정의 부산물은 제출용에서 빼고 data/processed/최종_순위.csv에만 둔다.
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

# (영문 컬럼명, 값, 데이터 타입, 타겟 여부, 비고) -- 이 순서가 CSV 열 순서
SPEC = [
    ("sigungu_code", df[KEY], "String", "X", "시군구 코드(5자리 문자열, 2026년 행정구역). 읽을 때 문자열로 읽어야 앞자리 0이 유지됨. 이름으로 병합 금지(서구·동구·고성군 등 중복)"),
    ("sido_name", df.시도명, "String", "X", "시도 이름(예: 전남광주통합특별시, 경기도)"),
    ("sigungu_name", df.시군구명, "String", "X", "시군구 이름. 구가 있는 시는 구 단위(예: 수원시 영통구), 2026년 개편 신설 구 포함(화성 4개 구, 인천 제물포·영종·서해·검단 등)"),
    ("power_substation_headroom_mw_2029", df["용량MW_2029"], "Float64", "X", "전력 원값. 2029년 기준 변전소(22.9kV·154kV) 단일 접속 가능 여유 최댓값, 단위 MW. 클수록 유리. 0은 여유 없음(실제 0)이며 결측은 빈칸"),
    ("power_score", df.점수_전력, "Float64", "X", "전력 점수 0~1. 평가 대상 245곳 안에서 power_substation_headroom_mw_2029를 min-max 정규화. 클수록 유리. 결측과 후보제외는 빈칸"),
    ("renewable_generation_mwh_2024", df.신재생_발전량_MWh, "Float64", "X", "신재생 원값. 2024년 재생에너지 발전량(MWh, 한국에너지공단 기초지자체별 보급 현황). 클수록 유리"),
    ("renewable_score", df.점수_신재생, "Float64", "X", "신재생 점수 0~1. log(1+발전량)을 min-max 정규화. 클수록 유리. 후보제외는 빈칸"),
    ("water_supply_headroom_m3_per_day", df.수자원_H, "Float64", "X", "수자원 원값. 최대급수일 기준 공급 여유(m³/일) = 자체 정수장 여유 + 연결된 광역 정수장 여유 + 이웃에서 받는 정수. 클수록 유리. 1GW 데이터센터의 하루 물 수요 기준값 4,080 m³/일(1GW × 24h × 공랭 WUE 0.17 L/kWh) 미만이면 용수선결 등급. 0은 여유 없음(실제 0)"),
    ("water_score", df.점수_수자원, "Float64", "X", "수자원 점수 0~1. log(1+공급 여유)를 min-max 정규화. 클수록 유리. 후보제외는 빈칸"),
    ("disaster_earthquake_score", df.지진_위험점수_1978_2025.where(df.재해_안전도.notna()), "Float64", "X", "재해 세부 점수 1. 지진 위험 점수 0~1(클수록 위험, 재해 담당 산출, 1978~2025 지진 기록 기반). 안전도 계산 가중치 0.19. 연안 고위험(CO Red) 10곳은 빈칸"),
    ("disaster_flood_score", df.재해_홍수점수_사용.where(df.재해_안전도.notna()), "Float64", "X", "재해 세부 점수 2. 홍수 위험 점수 0~1(클수록 위험, 100년 빈도 하천 침수 예상 면적 기반). 안전도 계산 가중치 0.17. 연안 고위험(CO Red) 10곳은 빈칸"),
    ("disaster_landslide_score", df.산사태_위험점수_2016_2025.where(df.재해_안전도.notna()), "Float64", "X", "재해 세부 점수 3. 산사태 위험 점수 0~1(클수록 위험, 2016~2025 산사태 발생·피해 기반). 안전도 계산 가중치 0.13. 연안 고위험(CO Red) 10곳은 빈칸"),
    ("disaster_wildfire_score", df.산불_위험점수_2011_2025.where(df.재해_안전도.notna()), "Float64", "X", "재해 세부 점수 4. 산불 위험 점수 0~1(클수록 위험, 2011~2025 산불 발생·피해 기반). 안전도 계산 가중치 0.08. 연안 고위험(CO Red) 10곳은 빈칸"),
    ("disaster_safety_index", df.재해_안전도, "Float64", "X", "재해 안전도 0~1 = 1 − 위험지수. 위험지수 = (0.19×지진 + 0.17×홍수 + 0.13×산사태 + 0.08×산불)/0.57(위 4개 세부 점수를 가중합, 연안 CO Yellow는 ×1.2). 클수록 안전해 유리. 연안 CO Red 10곳은 후보제외라 빈칸"),
    ("disaster_score", df.점수_재해, "Float64", "X", "재해 점수 0~1. disaster_safety_index를 min-max 정규화. 클수록 유리. 후보제외는 빈칸"),
    ("cooling_free_cooling_hour_ratio", df["냉각_WSE가능비율"], "Float64", "X", "냉각 원값. 습구온도 12.8℃ 이하인 시간의 비율(외기만으로 냉각 가능한 시간), 2024~2025 평균. 클수록 유리. 기상청 ASOS·AWS 시간자료, 습구온도는 Stull(2011) 식"),
    ("cooling_score", df.점수_냉각, "Float64", "X", "냉각 점수 0~1. cooling_free_cooling_hour_ratio를 min-max 정규화. 클수록 유리. 후보제외는 빈칸"),
    ("suitability_score", df.수용점수, "Float64", "O", "타겟 변수. 5개 점수의 가중합 0~1(클수록 수용 여건이 좋음). 가중치(논문 AHP 종합중요도를 5개 피처로 재정규화): 전력 41.75%, 신재생 19.22%, 수자원 16.39%, 재해 12.97%, 냉각 9.68%. 전력이 결측인 3곳은 남은 피처로 가중치를 다시 나눈 값(참고용). 후보제외 11곳은 빈칸"),
    ("suitability_grade", df.등급, "String", "X", "등급: A(상위 20%), B(20~50%), C(하위 50%)는 용수 기준을 통과한 지역 안에서의 구간. 용수선결(공급 여유가 4,080 m³/일 미만), 판정보류(전력 결측으로 등급 미부여), 제외(후보제외)"),
    ("reinforcement_needs", df.보강항목, "String", "X", "보강이 필요한 항목. 전력(변전소 여유 0MW), 전력(여유 있는 지역 중 하위 25%), 용수(공급 여유 < 4,080), 신재생·재해·냉각(평가 대상 하위 25%). 없으면 빈칸"),
    ("excluded_reason", df.제외사유.replace("", np.nan), "String", "X", "후보제외 사유. 연안 고위험(CO Red) 10곳 또는 전력 분석 제외(울릉군) 1곳. 제외가 아니면 빈칸"),
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
assert data.loc[m, [c for c in data.columns if c.startswith("disaster_") and c.endswith("_score") and c != "disaster_score"]].notna().all().all()

data.to_csv(OUT / "aidc_siting_dataset.csv", index=False, encoding="utf-8-sig")

# ---------- 컬럼 정의 ----------
na = data.isna().sum()
rows = []
for i, (name, _, typ, tgt, note) in enumerate(SPEC, 1):
    rows.append({"변수 번호": i, "타겟 변수 여부": tgt, "데이터 타입": typ, "데이터 컬럼 명": name,
                 "비고": f"{note} [빈칸 {int(na[name])}곳]" if na[name] else note})
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
print(tpl[["sido_name", "sigungu_name", "suitability_grade", "suitability_score"]].to_string(index=False))
