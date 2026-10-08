# 냉각(기온) 피처: AI 데이터센터 하이브리드 냉각(칩 액체냉각 + 공랭/외기 보조) 정의 기반
#
# 지표 (Notion "냉각 피처 검수" 문서 + Fuzzy-AHP 선행논문(이기수·정준호 2026) 종합):
#   [1] WSE 가능시간 - 습구온도 12.8C 이하 시간수 (수측 이코노마이저 기준, ENERGY STAR)
#       -> 외기/냉각탑으로 공랭 잔여부하를 식힐 수 있는 시간. 클수록 유리
#   [2] 여름 극값시간 - 6~8월 중 건구 33C 이상 또는 습구 25C 이상 시간수 (ASHRAE 2021 근거)
#       -> 고온수 액체냉각을 냉각탑만으로 방열하기 어려운 시간. 작을수록 유리
#   습구온도는 원자료에 없어 기온+상대습도로 근사(Stull 2011, 오차 약 ±0.3C).
#
# 관측소 선택 규칙 (144개 후보 행마다):
#   1. ASOS(97개)와 AWS(약 550개)를 하나의 관측소 목록으로 합친다. 주소는 기상청 공식
#      관측지점정보 CSV 두 개(ASOS용, 전체용)의 '지점주소'를 파싱한다.
#   2. 후보 행과 같은 구·읍면동·시군 안의 관측소를 모은다(광역시 구·세종 읍면동도 자체 관측소 사용).
#   3. 여름철(6~8월) 관측이 기대시간의 90% 미만인 관측소(신설·장기결측)와 레이더 관측소를 뺀다.
#   4. 남은 관측소 중 지역 이름을 딴 관측소(읍내·시가지)를 먼저 고르고, 없으면 해발고도가 가장 낮은
#      곳을 고른다. 섬·곶·산꼭대기가 아니라 데이터센터가 들어설 평지 시가지 기후를 대표하기 위함
#      (최저고도만 쓰면 통영=사량도, 여수=소리도처럼 섬 관측소가 뽑히는 문제가 있었다).
#      분석 기간 중 3km 넘게 이전한 관측소(예: 임실강진 6.5km)는 두 장소 기록이 섞여 제외한다.
#   5. 자체 관측소가 없거나 가장 낮은 곳도 400m를 넘으면(예: 부산 서구=산 위 레이더뿐),
#      상위 시(광역시 전체·세종 전체·청주시·포항시)에서 같은 규칙으로 고르고 상위시값공유=True로 표시.
#      상위 시가 없는 일반 군(예: 산간 군)은 400m를 넘어도 그 군의 최저 관측소를 그대로 쓴다.
#
# 수치는 관측 결측을 보정하기 위해 관측시간 대비 비율로 계산한 뒤 연 기준으로 환산한다
# (월별 비율을 달력 시간으로 가중평균 -> 연평균_WSE가능시간 = 비율*8760, 연평균_여름극값시간 = 여름 비율*2208).

import calendar
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트

import numpy as np
import pandas as pd

RAW_ASOS_DIR = ROOT / "data" / "raw" / "cooling" / "asos"
RAW_AWS_DIR = ROOT / "data" / "raw" / "cooling" / "aws"
OUT_DIR = ROOT / "data" / "processed" / "cooling"
REGION_LIST_PATH = ROOT / "data" / "raw" / "reference" / "행정시군구_기준행.csv"
META_ASOS = ROOT / "data" / "raw" / "cooling" / "meta" / "META_관측지점정보_20260928185946.csv"
META_ALL = ROOT / "data" / "raw" / "cooling" / "meta" / "META_관측지점정보_20260928193429.csv"

WSE_THRESHOLD_WB = 12.8
EXTREME_DB_THRESHOLD = 33.0
EXTREME_WB_THRESHOLD = 25.0
SUMMER_MONTHS = (6, 7, 8)
SUMMER_HOURS_PER_YEAR = (30 + 31 + 31) * 24
MIN_SUMMER_COVERAGE = 0.9
MAX_REPRESENTATIVE_ELEVATION_M = 400.0
MAX_RELOCATION_KM = 3.0
# 엄격한 기준을 만족하는 관측소가 없을 때만 쓰는 완화 기준(선택사유에 표시)
RELAXED_SUMMER_COVERAGE = 0.7
RELAXED_RELOCATION_KM = 5.0

SIDO_MAP = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "전남광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라남도": "전남광주", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주",
}
METRO = {"서울", "부산", "대구", "인천", "대전", "울산"}
GWANGJU_GU = {"동구", "서구", "남구", "북구", "광산구"}


def parse_unit(addr: str) -> tuple[str, str]:
    """지점주소 -> (시도, 후보목록과 같은 표기의 구·군·시·읍면동 단위)."""
    parts = re.sub(r"^\(산지\)", "", str(addr)).split()
    tok0 = parts[0]
    if tok0 == "전남광주통합특별시":
        return "전남광주", parts[1]
    sido = SIDO_MAP.get(tok0, tok0)
    if len(parts) < 2:
        return sido, sido
    tok1 = parts[1]
    if sido in METRO or sido == "세종":
        return sido, tok1
    if tok1.endswith("시") and len(parts) > 2 and parts[2].endswith("구"):
        return sido, tok1 + parts[2]  # "포항시 남구" -> "포항시남구"
    return sido, tok1


def relocation_km(period_start: str) -> pd.Series:
    """분석 기간 중 관측소가 옮겨간 거리(km). 관측지점정보의 이력 행(종료일 있는 행)으로 계산한다.
    대부분의 이력 변경은 장비·표기 변경이라 이동 거리가 0이다."""
    hist = pd.concat([pd.read_csv(p, encoding="cp949", skiprows=1) for p in (META_ASOS, META_ALL)]).drop_duplicates()
    for c in ("위도", "경도"):
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist[pd.to_datetime(hist["종료일"].fillna("2099-01-01"), errors="coerce") > period_start]
    out = {}
    for stn, h in hist.sort_values("시작일").groupby("지점"):
        a, b = h.iloc[0], h.iloc[-1]
        out[int(stn)] = float(np.hypot((a["위도"] - b["위도"]) * 111, (a["경도"] - b["경도"]) * 89))
    return pd.Series(out)


def load_station_catalog() -> pd.DataFrame:
    frames = []
    for path, source in ((META_ASOS, "ASOS"), (META_ALL, "AWS")):
        df = pd.read_csv(path, encoding="cp949", skiprows=1)
        df = df[df["종료일"].isna()].drop_duplicates("지점")
        frames.append(pd.DataFrame({
            "STN": df["지점"].astype(int),
            "관측소명": df["지점명"],
            "관측소주소": df["지점주소"],
            "해발고도_m": pd.to_numeric(df["노장해발고도(m)"], errors="coerce"),
            "출처": source,
        }))
    asos, allmeta = frames
    allmeta = allmeta[~allmeta["STN"].isin(asos["STN"])]
    cat = pd.concat([asos, allmeta], ignore_index=True)
    units = cat["관측소주소"].map(parse_unit)
    cat["시도"] = units.map(lambda t: t[0])
    cat["단위"] = units.map(lambda t: t[1])
    return cat


def wet_bulb_stull(ta: pd.Series, rh: pd.Series) -> pd.Series:
    return (
        ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(ta + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def monthly_counts(path: Path, stn_col: str, exclude_stn: set[int] | None) -> pd.DataFrame:
    """월 파일 하나에서 관측소별 (유효관측시간, WSE시간, 극값시간)을 센다."""
    ym = re.search(r"_(\d{4})(\d{2})\.csv", path.name)
    year, month = int(ym.group(1)), int(ym.group(2))
    df = pd.read_csv(path, usecols=["TM", stn_col, "TA", "HM"], dtype=str)
    df = df[df["TM"].str.match(r"^20\d{10}$", na=False)]
    df["STN"] = pd.to_numeric(df[stn_col], errors="coerce")
    df["TA"] = pd.to_numeric(df["TA"], errors="coerce")
    df["HM"] = pd.to_numeric(df["HM"], errors="coerce")
    df = df.dropna(subset=["STN"]).drop_duplicates(["TM", "STN"])
    df = df[df["TA"].between(-40, 50) & df["HM"].between(0, 100)]
    df["STN"] = df["STN"].astype(int)
    if exclude_stn:
        df = df[~df["STN"].isin(exclude_stn)]
    wb = wet_bulb_stull(df["TA"], df["HM"])
    df["wse"] = wb <= WSE_THRESHOLD_WB
    df["ext"] = (df["TA"] >= EXTREME_DB_THRESHOLD) | (wb >= EXTREME_WB_THRESHOLD)
    df["sat"] = df["HM"] >= 99.5
    g = df.groupby("STN").agg(obs=("TM", "size"), wse=("wse", "sum"), ext=("ext", "sum"), sat=("sat", "sum")).reset_index()
    g["year"], g["month"] = year, month
    return g


def _files(d: Path, prefix: str, years: list[int]) -> list[Path]:
    # 저장소에는 월별 CSV를 gzip(.csv.gz)으로 압축해 두었다. pandas가 그대로 읽는다.
    files = sorted(d.glob(f"{prefix}_*.csv")) + sorted(d.glob(f"{prefix}_*.csv.gz"))
    return [p for p in files if int(re.search(r"_(\d{4})\d{2}\.csv", p.name).group(1)) in years]


def load_station_counts(asos_ids: set[int], years: list[int]) -> pd.DataFrame:
    parts = [monthly_counts(p, "STN", None) for p in _files(RAW_ASOS_DIR, "asos_hourly", years)]
    # AWS 응답(awsh.php)에는 ASOS 지점도 섞여 오므로, ASOS 지점은 ASOS 원자료만 쓴다.
    parts += [monthly_counts(p, "AWS_ID", asos_ids) for p in _files(RAW_AWS_DIR, "aws_hourly", years)]
    return pd.concat(parts, ignore_index=True)


def _month_weighted_ratio(df: pd.DataFrame, num: str) -> pd.Series:
    """관측소·연도별로 월별 비율(num/obs)을 그 달의 달력 시간으로 가중평균한다.
    특정 계절에 결측이 몰린 관측소(예: 9~10월 결측)가 연간 합계 비율로 계산될 때 생기는
    계절 편향을 막기 위함."""
    df = df[df["obs"] > 0].copy()
    df["hours"] = [calendar.monthrange(y, m)[1] * 24 for y, m in zip(df["year"], df["month"])]
    df["wr"] = df[num] / df["obs"] * df["hours"]
    per_year = df.groupby(["STN", "year"]).agg(wr=("wr", "sum"), hours=("hours", "sum"))
    return per_year["wr"] / per_year["hours"]


def summarize(counts: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    counts = counts.copy()
    counts["summer"] = counts["month"].isin(SUMMER_MONTHS)
    s = counts.groupby("STN").agg(총관측시간=("obs", "sum"))
    s["여름관측시간"] = counts[counts["summer"]].groupby("STN")["obs"].sum()
    s["여름관측시간"] = s["여름관측시간"].fillna(0)
    s["여름관측커버리지"] = s["여름관측시간"] / (SUMMER_HOURS_PER_YEAR * len(years))
    # 습도 99.5% 이상 비율. 안개 잦은 분지이거나 센서가 높게 읽는 경우 높게 나오며,
    # 높으면 습구온도가 올라가 여름극값시간이 부풀려질 수 있어 검토용으로 남긴다.
    s["습도포화비율"] = (counts.groupby("STN")["sat"].sum() / s["총관측시간"]).round(4)

    wse_y = _month_weighted_ratio(counts, "wse").unstack("year")
    ext_y = _month_weighted_ratio(counts[counts["summer"]], "ext").unstack("year")
    s["WSE가능비율"] = wse_y.mean(axis=1)
    s["연평균_WSE가능시간"] = s["WSE가능비율"] * 8760
    s["여름극값비율"] = ext_y.mean(axis=1)
    s["연평균_여름극값시간"] = s["여름극값비율"] * SUMMER_HOURS_PER_YEAR
    for y in years:
        s[f"WSE가능시간_{y}"] = (wse_y.get(y) * 8760).round(0)
        s[f"여름극값시간_{y}"] = (ext_y.get(y) * SUMMER_HOURS_PER_YEAR).round(0)
    return s.reset_index()


def candidates_in(cat: pd.DataFrame, sido: str, unit: str) -> pd.DataFrame:
    same = cat[cat["시도"] == sido]
    if sido == "세종":
        return same
    unit = re.sub(r"\s+", "", unit)
    return same[(same["단위"] == unit) | (same["단위"].str.startswith(unit) & same["단위"].str.endswith("구"))]


def parent_candidates(cat: pd.DataFrame, sido: str, unit: str) -> pd.DataFrame | None:
    unit = re.sub(r"\s+", "", unit)
    if sido in METRO or sido == "세종":
        return cat[cat["시도"] == sido]
    if sido == "전남광주" and unit in GWANGJU_GU:
        return cat[(cat["시도"] == sido) & cat["단위"].isin(GWANGJU_GU)]
    m = re.match(r"^(.+?시)(.+구)$", unit)
    if m:
        return cat[(cat["시도"] == sido) & cat["단위"].str.startswith(m.group(1))]
    return None


def name_keys(sido: str, unit: str) -> list[str]:
    """관측소 이름과 대조할 지역명. '통영시'->['통영'], '청주시상당구'->['청주','상당'], '남구'->['남구'].
    시+구 지역은 시 이름을 먼저 본다(예: 상당구는 '상당'(미원면 산간)보다 '청주금천'(시가지)이 대표성 높음)."""
    keys = []
    unit = re.sub(r"\s+", "", unit)
    m = re.match(r"^(.+?)시(.+?)구$", unit)
    parts = [m.group(1), m.group(2)] if m else [re.sub(r"(시|군|구|읍|면|동)$", "", unit)]
    for p in parts:
        keys.append(p if len(p) >= 2 else unit)
    return keys


def pick(cands: pd.DataFrame, keys: list[str], min_cov: float = MIN_SUMMER_COVERAGE,
         max_reloc: float = MAX_RELOCATION_KM) -> tuple[pd.Series | None, str]:
    """1순위: 지역 이름을 딴 관측소(보통 읍내·시가지에 있어 섬·곶·산꼭대기보다 입지 후보지를 잘 대표함).
    2순위: 해발고도가 가장 낮은 관측소. 두 경우 모두 여름 관측 90% 이상, 레이더 제외,
    분석 기간 중 3km 넘게 이전한 관측소 제외, 400m 이하 조건을 먼저 적용한다."""
    ok = cands[
        (cands["여름관측커버리지"] >= min_cov)
        & ~cands["관측소명"].astype(str).str.contains(r"\(레\)")
        & cands["해발고도_m"].notna()
        & (cands["이동km"] <= max_reloc)
    ]
    if ok.empty:
        return None, ""
    # 주소에 "(산지)"가 붙은 산지 관측소는 다른 후보가 있으면 쓰지 않는다(예: 속초 설악동)
    lowland = ok[~ok["관측소주소"].astype(str).str.startswith("(산지)")]
    if not lowland.empty:
        ok = lowland
    low = ok[ok["해발고도_m"] <= MAX_REPRESENTATIVE_ELEVATION_M]
    order = ["해발고도_m", "총관측시간"]
    names = low["관측소명"].astype(str)
    for k in keys:
        # 이름이 정확히 같은 관측소가 먼저(예: 홍성 > 홍성죽도(섬)), 없으면 이름에 지역명이 들어간 관측소
        for named in (low[names == k], low[names.str.contains(k, regex=False)]):
            if not named.empty:
                return named.sort_values(order, ascending=[True, False]).iloc[0], "지역명 대표 관측소"
    return ok.sort_values(order, ascending=[True, False]).iloc[0], "최저고도 관측소"


def choose_station(cat: pd.DataFrame, sido: str, unit: str) -> tuple[pd.Series | None, bool, str]:
    keys = name_keys(sido, unit)
    own_c = candidates_in(cat, sido, unit)
    parent_c = parent_candidates(cat, sido, unit)
    ok = lambda st: st is not None and st["해발고도_m"] <= MAX_REPRESENTATIVE_ELEVATION_M

    own, how = pick(own_c, keys)
    if ok(own):
        return own, False, f"자체 {how}"
    # 자체 관측소가 기준(여름 90% 이상, 이전 3km 이하)에 못 미치면, 상위 시 값을 빌리기 전에
    # 완화 기준(여름 70% 이상, 이전 5km 이하)으로 자체 관측소를 한 번 더 찾는다.
    own2, how2 = pick(own_c, keys, RELAXED_SUMMER_COVERAGE, RELAXED_RELOCATION_KM)
    if ok(own2):
        return own2, False, f"자체 {how2}(기준 완화: 여름관측 {own2['여름관측커버리지']:.0%}, 이전 {own2['이동km']:.1f}km)"
    par, phow = pick(parent_c, keys) if parent_c is not None else (None, "")
    if ok(par):
        reason = "자체 관측소 없음" if own is None and own2 is None else f"자체 관측소가 {own['해발고도_m']:.0f}m 고지대" if own is not None else "자체 관측소가 기준 미달"
        return par, True, f"{reason} -> 상위 시 {phow}"
    par2, phow2 = pick(parent_c, keys, RELAXED_SUMMER_COVERAGE, RELAXED_RELOCATION_KM) if parent_c is not None else (None, "")
    if ok(par2):
        return par2, True, f"자체 관측소 없음 -> 상위 시 {phow2}(기준 완화)"
    if own is not None:
        return own, False, f"자체 {how}(지역 자체가 고지대)"
    return None, False, "관측소 없음"


def build(years: list[int]) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = pd.read_csv(REGION_LIST_PATH, encoding="utf-8-sig", dtype=str,
                          usecols=["시군구코드", "개편전코드", "시도", "시군구"]).fillna("")
    cat = load_station_catalog()
    counts = load_station_counts(set(cat.loc[cat["출처"] == "ASOS", "STN"]), years)
    cat = cat.merge(summarize(counts, years), on="STN", how="inner")
    cat["이동km"] = cat["STN"].map(relocation_km(f"{min(years)}-01-01")).fillna(0.0)

    rows = []
    for _, r in regions.iterrows():
        st, shared, reason = choose_station(cat, r["시도"], r["시군구"])
        row = {"시군구코드": r["시군구코드"], "개편전코드": r["개편전코드"], "시도": r["시도"], "시군구": r["시군구"],
               "상위시값공유": shared, "선택사유": reason}
        if st is not None:
            row.update(st.drop(labels=["시도", "단위"]).to_dict())
        rows.append(row)
    result = pd.DataFrame(rows)

    year_cols = [f"{p}_{y}" for p in ("WSE가능시간", "여름극값시간") for y in years]
    cols = [
        "시군구코드", "개편전코드", "시도", "시군구", "상위시값공유", "STN", "관측소명", "출처", "해발고도_m",
        "관측소주소", "이동km",
        "총관측시간", "여름관측시간", "여름관측커버리지", "습도포화비율",
        "연평균_WSE가능시간", "WSE가능비율", "연평균_여름극값시간", "여름극값비율",
        *year_cols, "선택사유",
    ]
    result["기준기간"] = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])

    def quality(r):
        # 값 자체는 바꾸지 않고 해석 시 주의할 행만 표시한다. 여러 개면 +로 연결.
        f = []
        if r["여름관측커버리지"] < 0.95:
            f.append("커버리지낮음")
        if "기준 완화" in str(r["선택사유"]):
            f.append("기준완화")
        if "고지대" in str(r["선택사유"]):
            f.append("고지대")
        return "+".join(f) or "정상"

    result["품질플래그"] = result.apply(quality, axis=1)
    cols = [*cols, "품질플래그", "기준기간"]
    result = result[cols]
    for c in ("연평균_WSE가능시간", "연평균_여름극값시간"):
        result[c] = result[c].round(0)
    for c in ("WSE가능비율", "여름극값비율", "여름관측커버리지"):
        result[c] = result[c].round(4)
    result["STN"] = result["STN"].astype("Int64")

    result.to_csv(OUT_DIR / "냉각_피처_시간기반.csv", index=False, encoding="utf-8-sig")

    print(f"기간: {years}, 관측소 후보 {len(cat)}개")
    print(f"매칭: {result['STN'].notna().sum()}/{len(result)}, 상위시값공유: {int(result['상위시값공유'].sum())}")
    print(result["출처"].value_counts().to_string())
    print(result["선택사유"].value_counts().to_string())
    print(result["품질플래그"].value_counts().to_string())
    print("\n고도 상위 10:")
    print(result.nlargest(10, "해발고도_m")[["시도", "시군구", "관측소명", "해발고도_m", "선택사유"]].to_string())
    return result


if __name__ == "__main__":
    import sys
    build([int(y) for y in sys.argv[1:]] or [2024, 2025])
