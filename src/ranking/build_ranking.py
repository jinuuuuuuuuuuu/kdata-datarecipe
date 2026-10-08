"""AI 데이터센터 수용 진단: 5개 피처 결합 -> 시군구별 수용 점수, 등급, 보강 항목.

절차
1) 시군구코드로 5개 피처 병합(256행)
2) 후보제외: 전력 '제외'(울릉) + 연안 CO Red(재해 후보제외)
3) 평가 대상(제외 뺀 행)에서 피처별 min-max 재정규화(클수록 유리)
     전력 = 전력피쳐(변전소 여유 MW/895), 신재생 = log(1+발전량), 수자원 = log(1+H), 재해 = 안전도, 냉각 = WSE 가능비율
4) 가중합: 논문(이기수·정준호 2026) AHP 종합중요도를 5개 피처로 재정규화. 결측 피처는 가중치를 남은 피처로 재정규화(방법 A)
5) 등급: 판정보류(전력 등 피처 결측, 점수는 참고로만 산출) / 용수 선결(H < D=4,080) / 나머지에서 A 상위 20%, B 20~50%, C 하위 50%
6) 보강 항목: 전력 여유 0MW, 용수(H<D), 그 밖에 평가 대상 하위 25% 피처
7) 민감도: Fuzzy-AHP 가중치, 동일 가중치, 신재생 대체배분, D 400/33,840
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
REPO = ROOT / "data" / "processed"
# 전력 피처는 전력 담당 팀원이 올리는 파일(data/raw/power/)을 쓴다. 경로를 인자로 줄 수도 있다.
POWER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "power" / "전력_최종피쳐_공유용_v4_20261006.csv"
OUT = REPO
if not POWER.exists():
    raise SystemExit(f"전력 피처 파일이 없습니다: {POWER} (전력 담당 팀원이 올린 파일 경로를 인자로 주세요)")
KEY = "시군구코드"
D_BASE = 4080.0

# 논문 표 5·6 종합중요도 -> 5개 피처(냉각환경 적합성은 세부요인으로 수자원/냉각 분리)
AHP = {"전력": 0.2420, "신재생": 0.1114, "수자원": 0.0528 + 0.0422, "재해": 0.0752, "냉각": 0.0282 + 0.0279}
FUZZY = {"전력": 0.3667, "신재생": 0.0798, "수자원": 0.0528 + 0.0412, "재해": 0.0467, "냉각": 0.0213 + 0.0130}
EQUAL = {k: 1.0 for k in AHP}
FEATS = list(AHP)
CAPITAL = {"서울", "인천", "경기"}


def read(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", dtype={KEY: str}, **kw)


pw = read(POWER)[[KEY, "시도명", "시군구명", "분류", "전력피쳐", "용량MW_2029", "결정전압", "수용가능_0.5GW"]]
wc = read(REPO / "수자원_냉각_통합.csv")[[KEY, "시도", "수자원_H", "냉각_WSE가능비율", "냉각_여름극값비율", "냉각_품질플래그"]]
rn = read(REPO / "신재생_피처.csv")[[KEY, "신재생_발전량_MWh", "신재생_발전량_대체배분_MWh", "신재생_출처", "신재생_추정", "신재생_배분_신뢰도"]]
dz = read(REPO / "재해_피처.csv")[[KEY, "재해_안전도", "재해_후보제외", "재해_홍수대체", "연안_Flag"]]
df = pw.merge(wc, on=KEY, validate="1:1").merge(rn, on=KEY, validate="1:1").merge(dz, on=KEY, validate="1:1")
assert len(df) == 256 and df[KEY].is_unique

# ---------- 후보제외 ----------
df["후보제외"] = (df.분류 == "제외") | df.재해_후보제외
df["제외사유"] = np.select([df.분류 == "제외", df.재해_후보제외], ["전력 분석 제외(울릉)", "연안 고위험(CO Red)"], "")
P = ~df.후보제외


def minmax(x, mask, bounds_mask=None):
    b = x[mask & (bounds_mask if bounds_mask is not None else True)]
    return ((x - b.min()) / (b.max() - b.min())).clip(0, 1).where(mask)


def feature_scores(d, ren_col="신재생_발전량_MWh"):
    s = pd.DataFrame(index=d.index)
    s["전력"] = minmax(d.전력피쳐, P)
    # 신재생 범위는 KEA 원자료가 직접 대응되는 행에서 잡고 배분 추정치는 잘라 범위를 흔들지 않게 한다
    s["신재생"] = minmax(np.log1p(d[ren_col].fillna(d.신재생_발전량_MWh)), P, ~d.신재생_추정)
    s["수자원"] = minmax(np.log1p(d.수자원_H), P)
    s["재해"] = minmax(d.재해_안전도, P)
    s["냉각"] = minmax(d.냉각_WSE가능비율, P)
    return s


def total(s, w):
    W = pd.Series(w)
    num = (s[FEATS] * W).sum(axis=1, min_count=1)
    den = s[FEATS].notna().mul(W).sum(axis=1)
    return (num / den).where(P)


def grade(score, h, d=D_BASE):
    g = pd.Series("", index=score.index, dtype=object)
    g[~P] = "제외"
    # 결측 피처가 있으면 남은 피처로 가중치를 재정규화한 점수가 부풀 수 있어 등급을 매기지 않는다
    hold = P & S[FEATS].isna().any(axis=1)
    g[hold] = "판정보류"
    water_fail = P & ~hold & (h < d)
    g[water_fail] = "용수선결"
    ok = P & ~hold & ~water_fail
    pct = score[ok].rank(ascending=False, pct=True)
    g[pct.index] = np.select([pct <= 0.2, pct <= 0.5], ["A", "B"], "C")
    return g


S = feature_scores(df)
for f in FEATS:
    df[f"점수_{f}"] = S[f]
df["수용점수"] = total(S, AHP)
df["부분점수"] = P & S[FEATS].isna().any(axis=1)
df["결측피처"] = S[FEATS].isna().apply(lambda r: ",".join(r.index[r]), axis=1).where(P & df.부분점수, "")
HOLD = P & S[FEATS].isna().any(axis=1)
df["전체순위"] = df.수용점수.where(~HOLD).rank(ascending=False, method="min")  # 판정보류는 순위에서 뺀다
df["등급"] = grade(df.수용점수, df.수자원_H)
okmask = df.등급.isin(["A", "B", "C"])
df["등급내순위"] = df.수용점수.where(okmask).rank(ascending=False, method="min")

# ---------- 보강 항목 / 강점 ----------
q25 = {f: S[f][P].quantile(0.25) for f in FEATS}
# 전력은 0MW가 평가 대상의 1/3을 넘어 하위 25%가 모두 0이므로, 여유가 있는 지역 안에서 하위 25%를 따로 잡는다
q25_power_nonzero = S["전력"][P & (df.용량MW_2029 > 0)].quantile(0.25)
q75 = {f: S[f][P].quantile(0.75) for f in FEATS}


def needs(i):
    if not P[i]:
        return ""
    out = []
    if df.at[i, "용량MW_2029"] == 0:
        out.append("전력(변전소 여유 0MW)")
    elif pd.notna(S.at[i, "전력"]) and S.at[i, "전력"] <= q25_power_nonzero:
        out.append("전력")
    if df.at[i, "수자원_H"] < D_BASE:
        out.append("용수(H<4,080)")
    for f in ("신재생", "재해", "냉각"):
        if pd.notna(S.at[i, f]) and S.at[i, f] <= q25[f]:
            out.append(f)
    return ", ".join(out)


df["보강항목"] = [needs(i) for i in df.index]
df["강점"] = [", ".join(f for f in FEATS if P[i] and pd.notna(S.at[i, f]) and S.at[i, f] >= q75[f]) for i in df.index]
df["수도권"] = df.시도.isin(CAPITAL)
df["전력_0.5GW수용"] = df["수용가능_0.5GW"].astype("boolean")  # 전력 담당 판정 그대로


def caution(i):
    c = []
    if df.at[i, "부분점수"]:
        c.append(f"결측:{df.at[i, '결측피처']}")
    if df.at[i, "신재생_추정"]:
        c.append(f"신재생 배분추정({df.at[i, '신재생_배분_신뢰도']})")
    if df.at[i, "재해_홍수대체"]:
        c.append("홍수 대체")
    if df.at[i, "냉각_품질플래그"] != "정상":
        c.append(f"냉각 {df.at[i, '냉각_품질플래그']}")
    return "; ".join(c)


df["해석주의"] = [caution(i) if P[i] else "" for i in df.index]

# ---------- 민감도 ----------
sens = {}
sens["Fuzzy-AHP"] = total(S, FUZZY)
sens["동일가중"] = total(S, EQUAL)
sens["신재생대체배분"] = total(feature_scores(df, "신재생_발전량_대체배분_MWh"), AHP)
for k, v in sens.items():
    df[f"민감도_순위_{k}"] = v.where(~HOLD).rank(ascending=False, method="min")
    df[f"민감도_등급_{k}"] = grade(v, df.수자원_H)
for d in (400.0, 33840.0):
    df[f"민감도_등급_D{int(d)}"] = grade(df.수용점수, df.수자원_H, d)

w = pd.Series(AHP) / sum(AHP.values())
cols = [KEY, "시도명", "시군구명", "등급", "전체순위", "등급내순위", "수용점수", "부분점수", "보강항목", "강점", "해석주의",
        *[f"점수_{f}" for f in FEATS], "용량MW_2029", "수자원_H", "신재생_발전량_MWh", "재해_안전도", "냉각_WSE가능비율",
        "전력_0.5GW수용", "후보제외", "제외사유", "수도권",
        *[c for c in df.columns if c.startswith("민감도_")]]
res = df[cols].sort_values(["후보제외", "전체순위"], na_position="last").reset_index(drop=True)
res.to_csv(OUT / "최종_순위.csv", index=False, encoding="utf-8-sig")

# ---------- 요약 출력 ----------
print("가중치(AHP 재정규화):", (w * 100).round(2).to_dict())
print("가중치(Fuzzy 재정규화):", (pd.Series(FUZZY) / sum(FUZZY.values()) * 100).round(2).to_dict())
print("등급 분포:", df.등급.value_counts().to_dict(), "| 부분점수:", int(df.부분점수.sum()))
sd = S[P].std().round(3).to_dict()
eff = (S[P].std() * w)
print("피처 표준편차:", sd, "| 실효 영향(%):", (eff / eff.sum() * 100).round(1).to_dict())
R = P & ~HOLD
base_r = df.수용점수.where(R).rank(ascending=False)
top20 = set(base_r[R].nsmallest(20).index)
for k, v in sens.items():
    r = v.where(R).rank(ascending=False)
    t = set(r[R].nsmallest(20).index)
    chg = (df[f"민감도_등급_{k}"] != df.등급)[P].sum()
    print(f"민감도 {k}: 순위상관 {base_r[R].corr(r[R], method='spearman'):.3f} | 상위20 겹침 {len(top20 & t)}/20 | 등급 바뀐 행 {chg}")
for d in (400, 33840):
    print(f"D={d}: 용수선결 {int((df[f'민감도_등급_D{d}'] == '용수선결').sum())}행 (기본 {int((df.등급 == '용수선결').sum())})")
show = ["전체순위", "시도명", "시군구명", "등급", "수용점수", *[f"점수_{f}" for f in FEATS], "보강항목", "해석주의"]
print("\n[상위 25]")
print(res[~res.후보제외].head(25)[show].round(3).to_string(index=False))
print("\n[확정 부지 검증: 세종·동해·울산]")
v = res[res[KEY].isin(["36110", "51170", "31110", "31140", "31170", "31200", "31710"])]
print(v[show].round(3).to_string(index=False))
