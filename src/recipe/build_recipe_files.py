"""레시피 제출용 파일 3개 생성: 데이터셋 CSV, 컬럼정의 CSV, 데이터 템플릿 예시 CSV.

입력: data/processed/ (최종_순위.csv, 수자원_냉각_통합.csv, 재해_피처.csv, 신재생_피처.csv)
출력: recipe/
  aidc_siting_dataset.csv            256행 x 32열 (타겟 변수: suitability_score)
  aidc_siting_column_definition.csv  변수 번호 / 타겟 변수 여부 / 데이터 타입 / 데이터 컬럼 명 / 비고
  aidc_siting_template_example.csv   같은 컬럼, 등급별 대표 5행
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "recipe"
OUT.mkdir(exist_ok=True)
KEY = "시군구코드"


def read(name):
    return pd.read_csv(PROC / name, encoding="utf-8-sig", dtype={KEY: str})


rk = read("최종_순위.csv")
wc = read("수자원_냉각_통합.csv")[[KEY, "수자원_H_자체만", "수자원_탈락여부", "냉각_여름극값비율", "냉각_품질플래그"]]
dz = read("재해_피처.csv")[[KEY, "재해_위험지수", "재해_홍수대체"]]
rn = read("신재생_피처.csv")[[KEY, "신재생_추정", "신재생_배분_신뢰도"]]
df = rk.merge(wc, on=KEY, validate="1:1").merge(dz, on=KEY, validate="1:1").merge(rn, on=KEY, validate="1:1")
assert len(df) == 256 and df[KEY].is_unique

P = ~df.후보제외  # 평가 대상 245곳

# (영문 컬럼명, 원본 열 또는 값, 데이터 타입, 타겟 여부, 비고)
SPEC = [
    ("sigungu_code", df[KEY], "String", "X", "시군구 코드(5자리 문자열, 2026년 행정구역). 읽을 때 문자열로 읽어야 앞자리 0이 유지됨. 이름으로 병합 금지(서구·동구·고성군 등 중복)"),
    ("sido_name", df.시도명, "String", "X", "시도 이름(예: 전남광주통합특별시, 경기도)"),
    ("sigungu_name", df.시군구명, "String", "X", "시군구 이름. 구가 있는 시는 구 단위(예: 수원시 영통구), 2026년 개편 신설 구 포함(화성 4개 구, 인천 제물포·영종·서해·검단 등)"),
    ("is_capital_region", df.수도권, "Boolean", "X", "수도권(서울·인천·경기) 여부. 해석용 구분이며 점수에는 쓰지 않음"),
    ("power_substation_headroom_mw_2029", df["용량MW_2029"], "Float64", "X", "전력 원값. 2029년 기준 변전소(22.9kV·154kV) 단일 접속 가능 여유 최댓값, 단위 MW. 클수록 유리. 0은 여유 없음(실제 0)이며 결측은 빈칸(6곳: 전력 분석 제외 1곳, 참고등급 5곳)"),
    ("power_can_host_500mw", df["전력_0.5GW수용"].map({True: True, False: False, "True": True, "False": False}), "Boolean", "X", "전력 담당 판정: 변전소 여유가 0.5GW 이상이면 True. 1GW 이상은 전국 0곳. 결측은 빈칸(6곳)"),
    ("renewable_generation_mwh_2024", df.신재생_발전량_MWh, "Float64", "X", "신재생 원값. 2024년 재생에너지 발전량(MWh, 한국에너지공단 기초지자체별 현황). 클수록 유리. 시 단위로만 제공되는 43개 구는 허가정보 설비용량 비율과 대형 발전소 소재지로 배분한 추정값(renewable_is_estimated 참고)"),
    ("water_supply_headroom_m3_per_day", df.수자원_H, "Float64", "X", "수자원 원값. 최대급수일 기준 공급 여유(m³/일) = 자체 정수장 여유 + 연결된 광역 정수장 여유 + 이웃에서 받는 정수. 낙관적 상한(같은 광역 정수장을 쓰는 지역이 여유를 중복으로 셈). 0은 여유 없음(실제 0)"),
    ("water_supply_headroom_own_plants_only_m3_per_day", df.수자원_H_자체만, "Float64", "X", "수자원 보수적 비교값. 자체 정수장 여유만(m³/일). 광역 정수장 연결을 제외해 동시 사용 중복이 없음"),
    ("disaster_risk_index", df.재해_위험지수, "Float64", "X", "재해 위험지수 0~1(클수록 위험). (0.19×지진 + 0.17×홍수 + 0.13×산사태 + 0.08×산불)/0.57, 연안 CO Yellow는 ×1.2. 연안 CO Red 10곳은 후보제외라 빈칸"),
    ("disaster_safety_index", df.재해_안전도, "Float64", "X", "재해 안전도 = 1 − disaster_risk_index. 클수록 유리. 후보제외 10곳은 빈칸"),
    ("cooling_free_cooling_hour_ratio", df["냉각_WSE가능비율"], "Float64", "X", "냉각 원값. 습구온도 12.8℃ 이하인 시간의 비율(외기만으로 냉각 가능한 시간), 2024~2025 평균. 클수록 유리. 기상청 ASOS·AWS 시간자료, 습구온도는 Stull(2011) 식"),
    ("cooling_summer_extreme_hour_ratio", df.냉각_여름극값비율, "Float64", "X", "참고용(점수에 쓰지 않음). 6~8월 중 건구온도 33℃ 이상 또는 습구온도 25℃ 이상인 시간의 비율. 작을수록 폭염 부담이 적음"),
    ("power_score", df.점수_전력, "Float64", "X", "전력 점수 0~1. 평가 대상 245곳 안에서 power_substation_headroom_mw_2029를 min-max 정규화. 클수록 유리. 결측 6곳과 후보제외는 빈칸"),
    ("renewable_score", df.점수_신재생, "Float64", "X", "신재생 점수 0~1. log(1+발전량)을 min-max 정규화(범위는 한국에너지공단 원자료가 직접 대응되는 행 기준, 배분 추정치는 0~1로 자름). 후보제외는 빈칸"),
    ("water_score", df.점수_수자원, "Float64", "X", "수자원 점수 0~1. log(1+공급 여유)를 min-max 정규화. 후보제외는 빈칸"),
    ("disaster_score", df.점수_재해, "Float64", "X", "재해 점수 0~1. disaster_safety_index를 min-max 정규화. 후보제외는 빈칸"),
    ("cooling_score", df.점수_냉각, "Float64", "X", "냉각 점수 0~1. cooling_free_cooling_hour_ratio를 min-max 정규화. 후보제외는 빈칸"),
    ("suitability_score", df.수용점수, "Float64", "O", "타겟 변수. 5개 점수의 가중합 0~1(클수록 수용 여건이 좋음). 가중치(논문 AHP 종합중요도를 5개 피처로 재정규화): 전력 41.75%, 신재생 19.22%, 수자원 16.39%, 재해 12.97%, 냉각 9.68%. 결측 피처가 있는 3곳은 남은 피처로 가중치를 다시 나눈 값(참고용). 후보제외 11곳은 빈칸"),
    ("national_rank", df.전체순위, "Int64", "X", "수용 점수 내림차순 순위(동점은 같은 순위). 후보제외 11곳과 판정보류 3곳은 빈칸"),
    ("suitability_grade", df.등급, "String", "X", "등급: A(상위 20%), B(20~50%), C(하위 50%)는 용수 기준을 통과한 지역 안에서의 구간. 용수선결(공급 여유가 물 수요 기준값 4,080 m³/일 미만), 판정보류(피처 결측으로 등급 미부여), 제외(후보제외)"),
    ("water_demand_shortfall", df.수자원_탈락여부, "Boolean", "X", "공급 여유가 1GW 데이터센터의 하루 물 수요 기준값(4,080 m³/일 = 1GW × 24h × 공랭 WUE 0.17 L/kWh)보다 작으면 True. 점수에는 쓰지 않고 등급 구분에만 사용"),
    ("reinforcement_needs", df.보강항목, "String", "X", "보강이 필요한 항목. 전력(변전소 여유 0MW), 전력(여유 있는 지역 중 하위 25%), 용수(공급 여유 < 4,080), 신재생·재해·냉각(평가 대상 하위 25%). 없으면 빈칸"),
    ("strengths", df.강점, "String", "X", "강점 항목. 평가 대상 상위 25%인 피처. 없으면 빈칸"),
    ("is_candidate_excluded", df.후보제외, "Boolean", "X", "후보제외 여부. 전력 분석 제외 1곳(울릉군)과 연안 고위험(CO Red) 10곳, 총 11곳"),
    ("excluded_reason", df.제외사유.replace("", np.nan), "String", "X", "후보제외 사유. 연안 고위험(CO Red) 또는 전력 분석 제외(울릉)"),
    ("has_missing_feature", df.부분점수, "Boolean", "X", "평가 대상 중 피처가 결측인 지역이면 True(3곳: 전력 결측). 이 경우 등급은 판정보류"),
    ("renewable_is_estimated", df.신재생_추정, "Boolean", "X", "신재생 값이 시(구) 합계를 구에 배분한 추정값이면 True(43곳)"),
    ("renewable_estimate_confidence", df.신재생_배분_신뢰도, "String", "X", "신재생 배분 신뢰도(높음/중간/낮음/매우 낮음). 추정값이 아닌 행은 빈칸. 허가정보 설비용량 커버율과 대형 발전소 귀속 가정에 따라 구분"),
    ("flood_score_imputed", df.재해_홍수대체, "Boolean", "X", "홍수위험지도가 없어 홍수 점수를 대체값으로 채운 행이면 True(6곳). 국가·지방하천이 없는 5곳은 원자료 하위 5% 값(0.024), 하천은 있으나 지도가 없는 신안군은 25분위 값(0.196)"),
    ("cooling_data_quality_flag", df.냉각_품질플래그, "String", "X", "냉각 자료 품질: 정상, 커버리지낮음(여름 관측 90% 미만), 기준완화, 고지대(대표 관측소 고도 400m 초과) 중 하나 또는 '+'로 결합"),
    ("interpretation_notes", df.해석주의.replace("", np.nan), "String", "X", "해석 시 주의 사항(결측 피처, 신재생 배분 추정, 홍수 대체, 냉각 품질). 후보제외 행은 빈칸"),
]

names = [s[0] for s in SPEC]
assert len(set(names)) == len(names)
data = pd.DataFrame({s[0]: s[1].values for s in SPEC})

# 숫자 자리수 정리
for c in data.columns:
    if c.endswith("_score") or c in ("disaster_risk_index", "disaster_safety_index", "suitability_score"):
        data[c] = data[c].round(4)
    elif c in ("cooling_free_cooling_hour_ratio", "cooling_summer_extreme_hour_ratio"):
        data[c] = data[c].round(4)
    elif c.endswith("_mwh_2024"):
        data[c] = data[c].round(1)
    elif c.endswith("_m3_per_day"):
        data[c] = data[c].round(0)
for c, typ in ((s[0], s[2]) for s in SPEC):
    if typ == "Int64":
        data[c] = data[c].astype("Int64")
    elif typ == "Boolean":
        data[c] = data[c].astype("boolean")

# ---------- 검수 ----------
assert data.sigungu_code.str.fullmatch(r"\d{5}").all()
sc = [c for c in data.columns if c.endswith("_score")]
assert (data[sc].stack().dropna().between(0, 1)).all()
assert data.loc[data.is_candidate_excluded, "suitability_score"].isna().all()
assert (data.suitability_grade[data.suitability_grade.isin(["A", "B", "C"])].count() == 201)
assert data.loc[data.suitability_grade == "용수선결", "water_demand_shortfall"].all()
assert not data.loc[data.suitability_grade.isin(["A", "B", "C"]), "water_demand_shortfall"].any()

data.to_csv(OUT / "aidc_siting_dataset.csv", index=False, encoding="utf-8-sig")

# ---------- 컬럼 정의 ----------
na = data.isna().sum()
rows = []
for i, (name, _, typ, tgt, note) in enumerate(SPEC, 1):
    rows.append({"변수 번호": i, "타겟 변수 여부": tgt, "데이터 타입": typ, "데이터 컬럼 명": name,
                 "비고": f"{note} [빈칸 {int(na[name])}곳]" if na[name] else note})
pd.DataFrame(rows).to_csv(OUT / "aidc_siting_column_definition.csv", index=False, encoding="utf-8-sig")

# ---------- 템플릿 예시(등급별 대표 5행) ----------
d = data.copy()
pick = []
for grade in ("A", "B", "C"):
    cand = d[(d.suitability_grade == grade) & d.interpretation_notes.isna() & d.reinforcement_needs.notna()]
    pick.append(cand.iloc[len(cand) // 2].name if len(cand) else d[d.suitability_grade == grade].iloc[0].name)
pick.append(d[d.suitability_grade == "용수선결"].sort_values("national_rank").iloc[0].name)
pick.append(d[d.suitability_grade == "제외"].sort_values("sigungu_code").iloc[0].name)
tpl = d.loc[pick]
tpl.to_csv(OUT / "aidc_siting_template_example.csv", index=False, encoding="utf-8-sig")

print(f"데이터셋 {data.shape}, 컬럼정의 {len(rows)}행, 템플릿 {tpl.shape}")
print("등급 분포:", data.suitability_grade.value_counts().to_dict())
print("타겟 결측:", int(data.suitability_score.isna().sum()), "| 타겟 범위:", data.suitability_score.min(), data.suitability_score.max())
print(tpl[["sigungu_name", "suitability_grade", "suitability_score", "national_rank"]].to_string(index=False))
