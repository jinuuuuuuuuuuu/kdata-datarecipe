"""신재생 피처: KEA 기초지자체별 재생에너지 발전량(2024) -> 256 시군구(2026 행정구역).
- KEA 시군구 값이 그대로 대응되는 행은 직접 사용
- 도 소속 시(13개)의 구, 인천 신설구 4개는 KEA 시(구) 값을 에너지원별로 배분
    태양광(및 위치 자료 없는 에너지원): 도시별로 정한 설비용량 비율(허가정보 등)
    풍력: 풍력기 위치 / 대형 단일 발전소(수력·바이오·해양): 소재지 구에 귀속
    민감도: 대체 비율(한전 PPA 계약용량, 허가 건수 등)
- 점수 = (log1p(MWh) - min) / (max - min)
"""
import re
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트

DL = ROOT / "data" / "raw" / "renewable"
EXTRA = DL
OUT = ROOT / "data" / "processed"
REF = ROOT / "data" / "raw" / "reference" / "행정시군구_기준행.csv"
LEGAL = ROOT / "data" / "raw" / "reference" / "법정동코드 전체자료.txt"
YEAR_CUT = 2024
SRC = ["태양열", "태양광", "풍력", "수력", "해양", "지열", "수열", "바이오", "폐기물"]

ref = pd.read_csv(REF, dtype=str, encoding="utf-8-sig")
kea = pd.read_csv(DL / "한국에너지공단_기초지자체별 신재생에너지 보급 현황_20241231.csv", encoding="cp949", dtype=str)
kea.columns = ["광역", "기초", "에너지원", "toe", "MWh", "누적kW", "신규kW"]
kea["MWh"] = pd.to_numeric(kea.MWh.str.replace(",", "").str.strip(), errors="coerce")
kea["기초"] = kea.기초.str.replace(" ", "")

# ---------- 1) 직접 매칭 ----------
tot = kea[kea.에너지원 == "재생에너지"].copy()
tot["k"] = tot.광역 + "|" + tot.기초
ref["광역"] = ref.시도.replace({"전남광주": "전남"})
ref.loc[ref.시군구코드.isin(["12210", "12240", "12270", "12300", "12330"]), "광역"] = "광주"   # 2026 개편: 광주 5개 구
ref["기초"] = ref.시군구_공백제거.replace({"세종특별자치시": "세종"})
ref["k"] = ref.광역 + "|" + ref.기초
m = ref.merge(tot[["k", "MWh"]], on="k", how="left").rename(columns={"MWh": "신재생_발전량_MWh"})
m["신재생_출처"] = np.where(m.신재생_발전량_MWh.notna(), "KEA 시군구 직접", "")

# ---------- 2) 배분용 자료 ----------
# 발전사업 허가정보 (전국 표준데이터: 가동 + 설치연도 <= 2024)
nat = pd.read_csv(DL / "전국태양광발전소전기사업허가정보표준데이터.csv", dtype=str)
nat["cap"] = pd.to_numeric(nat.설비용량, errors="coerce")
nat = nat[(nat.가동상태구분명 == "정상가동") & (nat.설치연도.str[:4].astype(int) <= YEAR_CUT)].copy()
nat["addr"] = nat.소재지도로명주소.fillna("") + " " + nat.소재지지번주소.fillna("")
# 화성: 허가정보 주소(읍면동)를 2026 신설 4개 구로 연결(법정동 파일)
legal = pd.read_csv(LEGAL, sep="\t", encoding="cp949", dtype=str)
hw = {}
for nm in legal.법정동명:
    mm = re.match(r"경기도 화성시 (\S+구) (\S+[읍면동])$", nm)
    if mm:
        hw.setdefault(mm.group(2), set()).add(mm.group(1))
hw = {k: next(iter(v)) for k, v in hw.items() if len(v) == 1}


def hwaseong_gu(addr):
    mm = re.search(r"화성시 (\S+구)", addr)
    if mm:
        return mm.group(1)
    mm = re.search(r"화성시 (\S+[읍면동])", addr)
    return hw.get(mm.group(1)) if mm else None


hs = nat[nat.addr.str.contains("화성시")].copy()
hs["gu"] = hs.addr.map(hwaseong_gu)
assert hs.gu.notna().mean() > 0.9, hs.gu.notna().mean()
# 충북 허가현황(청주 포함)
cb = pd.read_csv(DL / "충청북도_태양광발전소_전기사업허가현황_20260722.csv", dtype=str, encoding="utf-8-sig")
cb["cap"] = pd.to_numeric(cb["설비용량(kW)"].str.replace(",", ""), errors="coerce")
cb["개시"] = pd.to_datetime(cb.사업개시일자, errors="coerce")
cb = cb[(cb.상태 == "사업개시") & (cb.개시 <= f"{YEAR_CUT}-12-31")].copy()
cb["addr"] = cb.설치장소
# 고양시 / 성남시 / 화성시 자체 자료
gy = pd.read_csv(EXTRA / "고양시_태양광발전사업현황_20240328.csv", encoding="cp949", dtype=str)
gy["cap"] = pd.to_numeric(gy["설비용량(Kw)"].str.replace(",", ""), errors="coerce")
gy = gy[(gy.가동상태구분명 == "정상가동") & (gy.설치연도.str[:4].astype(int) <= YEAR_CUT)].copy()
gy["addr"] = gy.소재지도로명주소.fillna("") + " " + gy.소재지지번주소.fillna("")
sn = pd.read_csv(EXTRA / "성남시_태양광발전소_현황_20260529.csv", dtype=str)
sn["cap"] = pd.to_numeric(sn["설비용량(KW)"], errors="coerce")
sn = sn[pd.to_datetime(sn.사업개시일, errors="coerce") <= f"{YEAR_CUT}-12-31"].copy()
sn["addr"] = sn.소재지도로명주소.fillna("") + " " + sn.소재지지번주소.fillna("")
hf = pd.read_csv(EXTRA / "화성시_태양광발전소_20211201.csv", encoding="cp949", dtype=str)
hf["gu"] = hf["설비 위치"].str.extract(r"화성시 (\S+구)")[0]
# 한전 PPA(태양광), 풍력기
ppa = pd.read_csv(DL / "한국전력공사_지역별_PPA_계약현황_20251231.csv", encoding="cp949", dtype=str)
ppa = ppa[ppa.발전원 == "태양광"].copy()
ppa["cap"] = pd.to_numeric(ppa.용량, errors="coerce")
# "5호미만제거"(계약 5건 미만) 마스킹 행은 0이 아니라 2건 x 전체 평균 계약용량으로 보정, 목록에 없는 행은 0
_avg = (ppa.cap.sum() / pd.to_numeric(ppa.개수, errors="coerce").sum())
ppa["cap"] = ppa.cap.fillna(2 * _avg)
wind = pd.read_csv(DL / "한국에너지공단_풍력기_위치정보_20251231.csv", encoding="cp949")


def norm(s):
    return s / s.sum() if s.sum() > 0 else pd.Series(1 / len(s), index=s.index)


def cap_by(df, tg):
    return pd.Series({k: df[df.addr.str.contains(pat, regex=False)]["cap"].sum() for k, pat in tg.items()}, dtype=float)


def count_by(df, tg):
    return pd.Series({k: float(df.addr.str.contains(pat, regex=False).sum()) for k, pat in tg.items()})


def ppa_by(tg):
    return pd.Series({k: ppa[ppa.시도구분 == pat].cap.sum() for k, pat in tg.items()}, dtype=float)


def ratio(kind, tg):
    if kind == "nat":
        return cap_by(nat, tg)
    if kind == "natcnt":
        return count_by(nat, tg)
    if kind == "cb":
        return cap_by(cb, tg)
    if kind == "gy":
        return cap_by(gy, tg)
    if kind == "sn":
        return cap_by(sn, tg)
    if kind == "ppa":
        return ppa_by(tg)
    if kind == "pooled":
        return cap_by(nat, tg) + ppa_by(tg)
    if kind == "hs":
        return pd.Series({k: hs[hs.gu == pat.split()[-1]].cap.sum() for k, pat in tg.items()}, dtype=float)
    if kind == "hfcnt":
        return pd.Series({k: float((hf.gu == pat.split()[-1]).sum()) for k, pat in tg.items()})
    if kind == "equal":
        return pd.Series(1.0, index=list(tg))
    raise ValueError(kind)


def G(name, prov, units, tg, primary, alt, trust, basis, big=None, big_alt=None):
    return dict(name=name, prov=prov, units=units, tg=tg, primary=primary, alt=alt, trust=trust, basis=basis,
                big=big or {}, big_alt=big_alt or {})


def tgt(city, gus):
    return {f"{city}{g}": f"{city} {g}" for g in gus}


GROUPS = [
    G("청주시", "충북", ["청주시"], tgt("청주시", ["상당구", "서원구", "흥덕구", "청원구"]), "cb", "ppa", "중간",
      "충북 허가현황 설비용량 + 수력(대청댐)=상당구", big={"수력": "청주시상당구"}),
    G("천안시", "충남", ["천안시"], tgt("천안시", ["동남구", "서북구"]), "nat", "ppa", "높음", "허가정보 설비용량"),
    G("포항시", "경북", ["포항시"], tgt("포항시", ["남구", "북구"]), "nat", "ppa", "높음", "허가정보 설비용량 + 풍력기 위치"),
    G("창원시", "경남", ["창원시"], tgt("창원시", ["의창구", "성산구", "마산합포구", "마산회원구", "진해구"]), "nat", "ppa", "높음",
      "허가정보 설비용량"),
    G("전주시", "전북", ["전주시"], tgt("전주시", ["완산구", "덕진구"]), "nat", "ppa", "중간",
      "허가정보 설비용량 + 바이오(전주페이퍼)=덕진구", big={"바이오": "전주시덕진구"}, big_alt={"바이오": "전주시완산구"}),
    G("수원시", "경기", ["수원시"], tgt("수원시", ["장안구", "권선구", "팔달구", "영통구"]), "nat", "ppa", "중간",
      "허가정보 설비용량(시 태양광 추정용량의 약 50% 커버)"),
    G("성남시", "경기", ["성남시"], tgt("성남시", ["수정구", "중원구", "분당구"]), "sn", None, "낮음",
      "성남시 자체 허가 31건(약 17% 커버), PPA는 마스킹"),
    G("안양시", "경기", ["안양시"], tgt("안양시", ["만안구", "동안구"]), "equal", None, "매우 낮음",
      "허가 4건·PPA 마스킹으로 비율 근거 없음 -> 균등(시 값이 작아 영향 제한)"),
    G("부천시", "경기", ["부천시"], tgt("부천시", ["원미구", "소사구", "오정구"]), "equal", None, "매우 낮음",
      "PPA 전부 마스킹, 허가 소수 -> 균등(시 값이 작아 영향 제한)"),
    G("안산시", "경기", ["안산시"], tgt("안산시", ["상록구", "단원구"]), "nat", "ppa", "중간",
      "허가정보 설비용량(약 95% 커버) + 해양(시화호 조력)=단원구", big={"해양": "안산시단원구"}, big_alt={"해양": "안산시상록구"}),
    G("고양시", "경기", ["고양시"], tgt("고양시", ["덕양구", "일산동구", "일산서구"]), "gy", "ppa", "중간",
      "고양시 자체 허가현황 설비용량(약 70% 커버)"),
    G("용인시", "경기", ["용인시"], tgt("용인시", ["처인구", "기흥구", "수지구"]), "pooled", "ppa", "낮음",
      "허가정보+PPA 합산(허가 약 11% 커버, 수지구 PPA 마스킹)"),
    G("화성시", "경기", ["화성시"], tgt("화성시", ["만세구", "효행구", "병점구", "동탄구"]), "hs", "hfcnt", "중간",
      "허가정보 설비용량(읍면동->신설구 연결), 대체=화성시 발전소 건수"),
    G("인천 중·동구", "인천", ["중구", "동구"], {"영종구": "인천광역시 영종구", "제물포구": "인천광역시 제물포구"}, "nat", "natcnt", "중간",
      "2026 개편으로 중구·동구가 영종구·제물포구로 재편 -> 합계를 허가정보 설비용량으로 배분"),
    G("인천 서구", "인천", ["서구"], {"서해구": "인천광역시 서해구", "검단구": "인천광역시 검단구"}, "nat", "natcnt", "낮음",
      "2026 개편으로 서구가 서해구·검단구로 재편 -> 허가정보 설비용량 + 바이오(수도권매립지 매립가스)=검단구",
      big={"바이오": "검단구"}, big_alt={"바이오": "서해구"}),
]


def kea_total(prov, u):
    return float(tot[(tot.광역 == prov) & (tot.기초 == u)].MWh.iloc[0])


def kea_src(prov, u, s):
    v = kea[(kea.광역 == prov) & (kea.기초 == u) & (kea.에너지원 == s)].MWh
    return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else 0.0


rows, log = {}, []
for g in GROUPS:
    tg = g["tg"]
    keys = list(tg)
    r1 = norm(ratio(g["primary"], tg))
    r2 = norm(ratio(g["alt"], tg)) if g["alt"] else r1
    wv = pd.Series({k: wind[wind.소재지.astype(str).str.contains(pat, regex=False)
                            | wind.소재지.astype(str).str.contains(pat.replace(" ", ""), regex=False)]["용량(MW)"].sum()
                    for k, pat in tg.items()})
    res = {n: pd.Series(0.0, index=keys) for n in ("a", "b", "c")}
    for s in SRC:
        v = sum(kea_src(g["prov"], u, s) for u in g["units"])
        for n, rr, big in (("a", r1, g["big"]), ("b", r2, g["big"]), ("c", r1, {**g["big"], **g["big_alt"]})):
            if s in big:
                res[n][big[s]] += v
            elif s == "풍력" and wv.sum() > 0:
                res[n] += v * wv / wv.sum()
            else:
                res[n] += v * rr
    city_tot = sum(kea_total(g["prov"], u) for u in g["units"])
    log.append((g["name"], city_tot, res["a"].sum(), res["b"].sum()))
    for k in keys:
        rows[(g["prov"], k)] = dict(
            a=res["a"][k], b=res["b"][k], c=res["c"][k] if g["big_alt"] else np.nan, r1=r1[k], r2=r2[k],
            src=f"{g['name']} 값을 구에 배분", basis=g["basis"], trust=g["trust"], has_alt=bool(g["alt"]))
for name, c, a, b in log:
    assert abs(a - c) <= 5 and abs(b - c) <= 5, (name, c, a, b)   # 배분 합계 = 원본 합계 (KEA 에너지원별 반올림 허용)

for (prov, key), v in rows.items():
    hit = m.index[(m.광역 == prov) & (m.기초 == key)]
    assert len(hit) == 1, (prov, key)
    i = hit[0]
    m.loc[i, "신재생_발전량_MWh"] = v["a"]
    m.loc[i, "신재생_발전량_대체배분_MWh"] = v["b"] if v["has_alt"] else np.nan
    m.loc[i, "신재생_발전량_대형발전소반대귀속_MWh"] = v["c"]
    m.loc[i, "신재생_배분_비율"] = v["r1"]
    m.loc[i, "신재생_배분_대체비율"] = v["r2"] if v["has_alt"] else np.nan
    m.loc[i, "신재생_출처"] = v["src"]
    m.loc[i, "신재생_배분_근거"] = v["basis"]
    m.loc[i, "신재생_배분_신뢰도"] = v["trust"]

assert m.신재생_발전량_MWh.notna().all(), m[m.신재생_발전량_MWh.isna()][["시군구코드", "시도", "시군구"]].to_string()
assert (m.신재생_출처 != "").all()


# ---------- 3) 점수 ----------
def score(x):
    # 최소·최대는 KEA 직접 값(관측) 행에서만 잡고, 배분 추정치가 정규화 범위를 흔들지 않도록 0~1로 자른다
    z = np.log1p(x)
    zo = z[m.신재생_출처 == "KEA 시군구 직접"]
    return ((z - zo.min()) / (zo.max() - zo.min())).clip(0, 1)


m["신재생_점수"] = score(m.신재생_발전량_MWh)
m["신재생_점수_대체배분"] = score(m.신재생_발전량_대체배분_MWh.fillna(m.신재생_발전량_MWh))
m["신재생_추정"] = m.신재생_출처.str.contains("배분")
pct = lambda s: s.rank(pct=True) * 100
m["신재생_방법의존"] = m.신재생_추정 & ((pct(m.신재생_점수) - pct(m.신재생_점수_대체배분)).abs() >= 20)
m["신재생_기준연도"] = 2024

cols = ["시군구코드", "시도", "시군구", "신재생_발전량_MWh", "신재생_점수", "신재생_출처", "신재생_추정", "신재생_배분_신뢰도",
        "신재생_배분_근거", "신재생_방법의존", "신재생_발전량_대체배분_MWh", "신재생_점수_대체배분",
        "신재생_발전량_대형발전소반대귀속_MWh", "신재생_배분_비율", "신재생_배분_대체비율", "신재생_기준연도"]
m[cols].to_csv(OUT / "신재생_피처.csv", index=False, encoding="utf-8-sig")

print("점수 있는 행:", m.신재생_점수.notna().sum(), "/ 256 | 추정(배분) 행:", int(m.신재생_추정.sum()), "| 방법의존:", int(m.신재생_방법의존.sum()))
print(m.신재생_배분_신뢰도.value_counts().to_dict())
print("화성 읍면동 연결률:", round(hs.gu.notna().mean(), 3), "| 신설구별 용량(MW):", (hs.groupby("gu").cap.sum() / 1000).round(1).to_dict())
t = m[m.신재생_추정 & m.광역.isin(["경기", "인천"])]
print(t[["시군구", "신재생_발전량_MWh", "신재생_점수", "신재생_발전량_대체배분_MWh", "신재생_점수_대체배분",
         "신재생_발전량_대형발전소반대귀속_MWh", "신재생_배분_신뢰도"]].round(3).to_string(index=False))
print(m.신재생_점수.describe().round(3).to_dict())
