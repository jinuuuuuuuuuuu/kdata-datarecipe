# 수자원 피처 재구축: 연평균 취수여유율 → 첨두일 공급가능 여유량(m3/일) 기반
#
# 시트 구조 (실측 확인 결과, 3-2-1 정수시설현황 기준):
#   col1(수도사업자) = "충청남도 태안군" 형태의 "시도 시군구" 문자열. 소계행("경기도","경기도 시",
#   "서울특별시" 등 시군구 없는 행)과 "한국수자원공사"(광역 전용 그룹)는 상세 데이터가 아니므로 제외.
#   col3(정수장명) = 해당 시군구가 쓰는 개별 정수장. "계"=사업자 합계행, "(광역)이름"=K-water 광역
#   정수장을 끌어쓴다는 연결 행(이 행 자체의 F9/F26은 0 — 실제 용량은 한국수자원공사 그룹의 동명
#   정수장 행에 있음), "(운휴)"/"(통합)"/"(폐쇄)"는 자체 시설의 특수 상태.
#
# 계산 로직:
#   1. 자체 정수 여유 H_own_T = 가동 중 정수장(운휴·폐쇄 제외)의 F9(설계시설용량) - F26(일최대생산량) 합
#      (3-1 가동 중 취수장으로 취수 측 상한 H_own_I도 계산해 min으로 캡. 단 원수를 수입하는 사업자는 캡 미적용)
#   2. 광역 배분 H_kw = 시군구별 "(광역)정수장명" 행에서 정수장명을 뽑아, 한국수자원공사 그룹의
#      동명 정수장 행(F9-F26=여유)을 찾아 그 정수장을 쓰는 모든 시군구의 실공급량(2-3-2) 비중으로 배분
#   3. H = max(H_own_T,0) [+min(H_own_I) 캡] + H_kw
#   4. U(가동률) = 가동 중 정수장의 일최대생산 합 / 설계용량 합
#   5. 물 수요 기준값 D=4,080 m3/일(1GW x 24h x 공랭 WUE 0.17 L/kWh)로 탈락·커버리지·Water 점수 산출, 민감도 D=400/33,840
#
# 한계:
#   - 법적 수리권(하천수 사용허가)은 반영하지 않음 — 설계시설용량 기준
#   - 2024년 단년도 자료만 사용
#   - "(광역)정수장" 연결 행에 실제 배분량이 없어, 실공급량(2-3-2) 비중으로 근사 배분함
#   - 세종시·광역시·청주·포항은 사업자가 시 단위 하나라 구·읍면동 세분 불가 -> 상위시값공유 컬럼으로 표시
#   - 국가수도기본계획 고시문의 "신규 확충 대상 지역"은 별도 플래그로만 추가 (점수 미반영)
#   - 공업용수 여유(3-2-2)는 참고 컬럼으로만 추가 (점수 미반영). K-water 공업용 정수장은 급수지역에
#     적힌 시군 수로 균등 배분한 근사값

import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트

import numpy as np
import pandas as pd

XLSX_PATH = ROOT / "data" / "raw" / "water" / "2024년_상수도통계_공표.xlsx"
# 정부 AIDC 메가프로젝트(2026-06-29) 부지 단위 0.5~1GW -> 1GW 기준. 첨두계수는 적용하지 않는다.
D_BASE = 1000 * 24 * 0.17  # m3/일: 1GW x 24h x 공랭 WUE 0.17 L/kWh(한국환경연구원) = 4,080
D_SENSITIVITY = {"D400": 400.0, "D33840": 1000 * 24 * 1.41}  # 정부 구두보고(폐쇄루프), 수랭 WUE 1.41(한국환경연구원)
REGION_LIST_PATH = ROOT / "data" / "raw" / "reference" / "행정시군구_기준행.csv"
OUT_DIR = ROOT / "data" / "processed" / "water"

SIDO_FULL = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시",
    "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시",
    "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도",
    "제주": "제주특별자치도",
}
METRO_SIDO_FULL = {"서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시"}
AGGREGATE_ROW_SUFFIXES = (" 시", " 군")  # "경기도 시" / "경기도 군" 형태의 소계행



def read_sheet_raw(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=sheet, header=None)


def symbol_col_map(raw: pd.DataFrame, symbol_row: int = 7) -> dict:
    syms = raw.iloc[symbol_row].tolist()
    mapping = {}
    for i, s in enumerate(syms):
        if isinstance(s, str) and s.strip():
            key = s.split("=")[0].strip().replace("\n", "")
            if key and key not in mapping:
                mapping[key] = i
    return mapping


# 제주특별자치도는 하위 시 구분 없이 도 전체가 사업자 하나(정수장 17곳)라 소계행이 아니다.
DO_ONLY_NAMES = {v for k, v in SIDO_FULL.items()
                 if v not in METRO_SIDO_FULL and v not in ("세종특별자치시", "제주특별자치도")}


def is_detail_utility_row(name: str) -> bool:
    """소계행("경기도","경기도 시" 등)과 한국수자원공사 그룹, 상위 집계행을 제외.
    광역시(부산광역시 등)·세종특별자치시는 그 자체가 실제 사업자 상세행이므로 제외하지 않는다."""
    if not isinstance(name, str):
        return False
    if name in ("(지자체+수공)", "(지자체)", "(특광역시)", "(특별자치시)", "한국수자원공사"):
        return False
    if name.endswith(AGGREGATE_ROW_SUFFIXES):
        return False
    if name in DO_ONLY_NAMES:
        return False
    return " " in name or name in METRO_SIDO_FULL or name in ("세종특별자치시", "제주특별자치도")


def split_sido_sigungu(name: str) -> tuple[str, str]:
    """'충청남도 태안군' -> ('충남','태안군'); '세종특별자치시' -> ('세종','세종특별자치시')."""
    full_to_short = {v: k for k, v in SIDO_FULL.items()}
    if name == "세종특별자치시":
        return "세종", "세종시"
    parts = name.split(" ", 1)
    if len(parts) == 2 and parts[0] in full_to_short:
        return full_to_short[parts[0]], parts[1]
    return name, name


def load_plant_table(xlsx_path: Path, sheet: str, cap_key: str, prod_key: str, util_rate_key: str | None) -> pd.DataFrame:
    raw = read_sheet_raw(xlsx_path, sheet)
    col = symbol_col_map(raw)
    body = raw.iloc[8:].copy()
    df = pd.DataFrame({
        "사업자원문": body[1],
        "시설명": body[3],
        "설계시설용량": pd.to_numeric(body[col[cap_key]], errors="coerce"),
        "일최대생산": pd.to_numeric(body[col[prod_key]], errors="coerce"),
    })
    if util_rate_key and util_rate_key in col:
        df["가동률_최대"] = pd.to_numeric(body[col[util_rate_key]], errors="coerce")
    else:
        df["가동률_최대"] = np.nan
    return df


def load_contracts(xlsx_path: Path) -> pd.DataFrame:
    """2-3-2 광역상수도 급수계약현황: 사용자(수요 시군)별 실공급량."""
    raw = read_sheet_raw(xlsx_path, "2-3-2 광역상수도 급수계약현황")
    col = symbol_col_map(raw)
    body = raw.iloc[8:].copy()
    # 열 위치: 3=지역구분(시도), 4=사용자명(C11), 5=권역(C12). 예전 코드는 C12를 사용자명으로 읽어
    # 권역명("수도권광역상수도")이 키가 되는 바람에 계약량이 어느 지역과도 맞지 않았다.
    df = pd.DataFrame({
        "지역구분": body[3],
        "사용자명": body[col["C11"]],
        "권역": body[col["C12"]],
        "실공급량_계": pd.to_numeric(body[col["C23"]], errors="coerce"),
    })
    df = df.dropna(subset=["사용자명", "지역구분"])
    df = df[~df["사용자명"].astype(str).str.fullmatch(r".*계", na=False)]
    return df


def load_balance_analysis(xlsx_path: Path) -> pd.DataFrame:
    raw = read_sheet_raw(xlsx_path, "4-1-1 총괄수량 수지분석")
    col = symbol_col_map(raw)
    body = raw.iloc[9:].copy()
    df = pd.DataFrame({
        "사업자원문": body[1],
        "광역수입량": pd.to_numeric(body[col["M39"]], errors="coerce") if "M39" in col else np.nan,
        "지방자급률": pd.to_numeric(body[col["M38"]], errors="coerce") if "M38" in col else np.nan,
    })
    return df.dropna(subset=["사업자원문"])


# 사업자·지역 키는 "시도|단위" 문자열로 만든다(이름만 쓰면 경기 광주시와 광주광역시, 강원·경남 고성군이
# 같은 키가 돼 값이 합쳐진다). 시도는 최신 개편을 반영해 광주·전남을 "전남광주"로 묶는다.
SIDO_SHORT_FROM_FULL = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
    "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남", "전북특별자치도": "전북",
    "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
    "전라남도": "전남광주", "광주광역시": "전남광주", "전남광주통합특별시": "전남광주",
}
WHOLE_METRO = {"서울", "부산", "대구", "인천", "대전", "울산", "세종", "제주"}  # 시·도 전체가 사업자 1개
GWANGJU_GU = {"동구", "서구", "남구", "북구", "광산구"}


def _unit_key(sido_short: str, unit: str) -> str:
    unit = re.sub(r"\s+", "", unit)
    m = re.match(r"^(.+?시)(.+구)$", unit)  # '수원시장안구' -> 모시 '수원시'
    if m:
        unit = m.group(1)
    return f"{sido_short}|{unit}"


def key_from_raw_name(name: str) -> str:
    """상수도통계 사업자명('경기도 수원시', '부산광역시', '전라남도 목포시')을 키로 바꾼다."""
    name = name.strip()
    if name == "광주광역시":
        return "전남광주|광주광역시"
    if name in SIDO_SHORT_FROM_FULL and SIDO_SHORT_FROM_FULL[name] in WHOLE_METRO:
        s_ = SIDO_SHORT_FROM_FULL[name]
        return f"{s_}|{s_}"
    sido_full, _, rest = name.partition(" ")
    return _unit_key(SIDO_SHORT_FROM_FULL.get(sido_full, sido_full), rest)


def region_utility_key(sido_short: str, unit: str) -> str:
    """후보 지역(기준행의 시도·시군구)이 속한 상수도 사업자 키."""
    if sido_short in WHOLE_METRO:
        return f"{sido_short}|{sido_short}"
    if sido_short == "전남광주" and re.sub(r"\s+", "", unit) in GWANGJU_GU:
        return "전남광주|광주광역시"
    return _unit_key(sido_short, unit)


def key_from_contract(sido_full: str, user: str) -> str:
    """2-3-2 계약 시트의 (지역구분=시도, 사용자명)을 키로 바꾼다."""
    s_ = SIDO_SHORT_FROM_FULL.get(str(sido_full).strip(), str(sido_full).strip())
    if s_ in WHOLE_METRO:
        return f"{s_}|{s_}"
    if str(sido_full).strip() == "광주광역시":
        return "전남광주|광주광역시"
    return _unit_key(s_, str(user))


# 국가수도기본계획(환경부고시 제2022-187호) 3쪽 "주요 신규 확충 계획"의 사업별 급수지역.
# 고시문 요약본에는 시군별 2035 과부족 수치가 없으므로 점수에는 반영하지 않고 플래그로만 쓴다.
# 표기가 모호한 지명은 시도를 명시했다(남강권1차의 '고성'=경남 고성, 한강하류6차의 '광주'=경기 광주).
SUPPLY_EXPANSION_PROJECTS = {
    "청주 하수재이용": [("충북", "청주")],
    "군산 하수재이용": [("전북", "군산")],
    "여수 하수재이용": [("전남", "여수")],
    "보성 지하수저류지": [("전남", "여수"), ("전남", "광양")],
    "광양 지하수저류지": [("전남", "여수"), ("전남", "광양")],
    "남한강3차": [("충북", "괴산"), ("충북", "음성"), ("경기", "안성"), ("충북", "진천")],
    "낙동강중부3차": [("경북", "김천"), ("경북", "구미"), ("경북", "칠곡")],
    "금호강1차": [("대구", "대구"), ("경북", "경산"), ("경북", "영천"), ("경북", "청도")],
    "남강권1차": [("경남", "진주"), ("경남", "통영"), ("경남", "사천"), ("경남", "거제"),
                ("경남", "고성"), ("경남", "남해"), ("경남", "하동")],
    "창원국가산단": [("경남", "창원")],
    "밀양나노융합국가산단": [("경남", "밀양")],
    "경남항공국가산단": [("경남", "진주"), ("경남", "사천")],
    "대청댐계통": [("대전", "대전"), ("세종", "세종"), ("충북", "청주"), ("충남", "천안"), ("충남", "예산")],
    "금강북부3차": [("세종", "세종"), ("충남", "보령"), ("충남", "당진"), ("충남", "청양"), ("충남", "예산"),
                ("충남", "태안"), ("충남", "부여"), ("충남", "서산"), ("충남", "홍성")],
    "금강남부3차": [("충남", "서천"), ("전북", "전주"), ("전북", "군산"), ("전북", "익산"), ("전북", "김제"),
                ("전북", "완주"), ("전북", "부안"), ("전북", "고창")],
    "영산강3차": [("전남", "목포"), ("전남", "장흥"), ("전남", "강진"), ("전남", "해남"), ("전남", "영암"),
               ("전남", "무안"), ("전남", "완도"), ("전남", "진도"), ("전남", "신안")],
    "충주댐Ⅲ": [("충북", "괴산"), ("충북", "음성"), ("경기", "안성"), ("충북", "진천")],
    "금산Ⅱ": [("충남", "금산"), ("전북", "진안")],
    "광양Ⅳ 공업": [("전남", "여수"), ("전남", "광양")],
}
WHOLE_CITY_SIDO = {"대구", "대전", "세종"}  # 사업 대상이 시 전체인 곳


def expansion_projects_for(sido: str, unit: str) -> list[str]:
    u = re.sub(r"\s+", "", unit)
    hits = []
    for project, targets in SUPPLY_EXPANSION_PROJECTS.items():
        for t_sido, t_name in targets:
            t_sido = "전남광주" if t_sido in ("전남", "광주") else t_sido
            if sido != t_sido:
                continue
            if sido in WHOLE_CITY_SIDO or u.startswith(t_name):
                hits.append(project)
                break
    return hits


def load_raw_water_import_by_key(xlsx_path: Path) -> pd.Series:
    """3-2-1 사업자 "계" 행의 F20(원수+침전수 수입량, m3/년)을 key별로 합산."""
    raw = read_sheet_raw(xlsx_path, "3-2-1 정수시설현황")
    col = symbol_col_map(raw)
    body = raw.iloc[8:]
    df = pd.DataFrame({
        "사업자원문": body[1], "시설명": body[3],
        "원수수입": pd.to_numeric(body[col["F20"]], errors="coerce").fillna(0),
    })
    df = df[(df["시설명"] == "계") & df["사업자원문"].map(is_detail_utility_row)]
    df["key"] = df["사업자원문"].map(key_from_raw_name)
    return df.groupby("key")["원수수입"].sum()


def load_industrial_water_by_key(xlsx_path: Path, token_index: dict) -> pd.DataFrame:
    """3-2-2 공업용 정수시설현황 -> key별 공업용수 첨두일 여유(m3/일). 참고용 컬럼(점수 미반영).
    지자체 정수장은 사업자 key에 그대로 귀속, K-water 공업용 정수장은 급수지역 텍스트에 적힌 시군에 균등 배분.
    (운휴)/(폐쇄) 시설은 생산 실적이 0이라 설계용량 전체가 여유로 잡히는 왜곡이 있어 제외한다."""
    raw = read_sheet_raw(xlsx_path, "3-2-2 공업용 정수시설현황")
    col = symbol_col_map(raw)
    body = raw.iloc[8:]
    df = pd.DataFrame({
        "사업자원문": body[1],
        "시설명": body[3].astype(str),
        "설계": pd.to_numeric(body[col["F9"]], errors="coerce").fillna(0),
        "일최대": pd.to_numeric(body[col["F26"]], errors="coerce").fillna(0),
        "급수지역": body[col["F36"]].astype(str),
    })
    df = df[(df["시설명"] != "계") & ~df["시설명"].str.match(r"^\((운휴|폐쇄|광역)\)")]
    df["여유"] = (df["설계"] - df["일최대"]).clip(lower=0)

    rows = []
    own = df[df["사업자원문"].map(is_detail_utility_row)]
    for _, r in own.iterrows():
        rows.append({"key": key_from_raw_name(r["사업자원문"]), "공업용수_여유": r["여유"], "시설": r["시설명"]})

    kw = df[df["사업자원문"] == "한국수자원공사"]
    for _, r in kw.iterrows():
        tokens = [t for t in re.split(r"[,\s]+", r["급수지역"]) if t]
        keys = []
        for i, t in enumerate(tokens):
            if re.search(r"(시|군)$", t) or i == 0:
                keys.extend(token_index.get(re.sub(r"(특별자치시|시|군)$", "", t), []))
        keys = list(dict.fromkeys(keys))
        for k in keys:
            rows.append({"key": k, "공업용수_여유": r["여유"] / len(keys), "시설": f"K-water {r['시설명']}"})

    out = pd.DataFrame(rows)
    return out.groupby("key").agg(
        공업용수_여유=("공업용수_여유", "sum"), 공업용수_시설=("시설", lambda s: ", ".join(s))
    ).reset_index()


def build(xlsx_path: Path, region_list_path: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = pd.read_csv(region_list_path, encoding="utf-8-sig", dtype=str).fillna("")
    regions["key"] = regions.apply(lambda r: region_utility_key(r["시도"], r["시군구"]), axis=1)
    # K-water 공업용수 급수지역 텍스트('평택시, 아산시…')의 시군 이름을 키로 바꾸는 색인
    token_index: dict[str, list[str]] = {}
    for _, r in regions.iterrows():
        base_unit = re.match(r"^(.+?시)", r["시군구"]) if " " in r["시군구"] else None
        stem = re.sub(r"(특별자치시|시|군|구)$", "", base_unit.group(1) if base_unit else r["시군구"])
        for nm in {stem, r["시도"] if r["시도"] in WHOLE_METRO else ""}:
            if nm and r["key"] not in token_index.setdefault(nm, []):
                token_index[nm].append(r["key"])

    plants = load_plant_table(xlsx_path, "3-2-1 정수시설현황", "F9", "F26", "F39")
    intakes = load_plant_table(xlsx_path, "3-1 취수시설현황", "E8", "E23", "E29")
    contracts = load_contracts(xlsx_path)
    balance = load_balance_analysis(xlsx_path)

    plants["is_detail"] = plants["사업자원문"].map(is_detail_utility_row)
    detail = plants[plants["is_detail"]].copy()
    detail["key"] = detail["사업자원문"].map(key_from_raw_name)

    # 자체 정수 여유는 "계" 합계행이 아니라 가동 중인 정수장만 더해 계산한다. "계" 행에는
    # (운휴)·(폐쇄) 정수장 용량이 포함돼, 쓰지 않는 시설이 여유로 잡히는 왜곡이 있었다
    # (예: 아산·나주·청양은 자체 여유 전부가 운휴 시설). (광역)·(정수수입)은 용량 0인 연결 표시.
    # (마을상수도)는 사업자 본 정수장과 별개인 소규모 급수 시설이고 "계" 합계에도 들어가지 않아 제외한다.
    inactive = detail["시설명"].astype(str).str.match(r"^\((광역|운휴|폐쇄|정수수입|마을상수도)\)")
    own_rows = detail[(detail["시설명"] != "계") & ~inactive].copy()
    # 참고용: 운휴·폐쇄 정수장의 설계용량(재가동 시 잠재 용량). 여유·가동률 계산에는 넣지 않는다.
    idle_plants = detail[detail["시설명"].astype(str).str.match(r"^\((운휴|폐쇄)\)")]
    idle_by_key = idle_plants.groupby("key")["설계시설용량"].sum()
    kw_link_rows = detail[detail["시설명"].astype(str).str.startswith("(광역)")].copy()
    kw_link_rows["정수장명"] = kw_link_rows["시설명"].str.replace("(광역)", "", regex=False)
    own_rows["H_own_T"] = own_rows["설계시설용량"] - own_rows["일최대생산"]

    # 원수·침전수를 외부(주로 K-water 광역원수)에서 사오는 사업자는 자체 취수장 용량이 정수 한도가
    # 아니므로 취수 측 상한을 걸지 않는다(청주·공주·영천 등에서 여유가 음수로 떨어지던 문제).
    raw_import = load_raw_water_import_by_key(xlsx_path)

    intakes["is_detail"] = intakes["사업자원문"].map(is_detail_utility_row)
    intake_detail = intakes[intakes["is_detail"]].copy()
    intake_detail["key"] = intake_detail["사업자원문"].map(key_from_raw_name)
    intake_inactive = intake_detail["시설명"].astype(str).str.match(r"^\((광역|운휴|폐쇄|마을상수도)\)")
    intake_own = intake_detail[(intake_detail["시설명"] != "계") & ~intake_inactive].copy()
    intake_own["H_own_I"] = intake_own["설계시설용량"] - intake_own["일최대생산"]

    # K-water 정수장 원천 데이터 (설계용량/생산량이 실제로 채워진 행) = "한국수자원공사" 그룹
    kw_source = plants[plants["사업자원문"] == "한국수자원공사"].copy()
    kw_source["정수장명"] = kw_source["시설명"]
    kw_source = kw_source[kw_source["정수장명"] != "계"]
    kw_source["여유"] = (kw_source["설계시설용량"] - kw_source["일최대생산"]).clip(lower=0)

    # 각 광역 정수장을 쓰는 시군구 목록(연결행에서 추출) + 실공급량 비중으로 배분
    contracts["key"] = [key_from_contract(a, b) for a, b in zip(contracts["지역구분"], contracts["사용자명"])]
    contract_by_key = contracts.groupby("key")["실공급량_계"].sum()

    kw_alloc_rows = []
    for plant_name, grp in kw_link_rows.groupby("정수장명"):
        plant_row = kw_source[kw_source["정수장명"] == plant_name]
        if plant_row.empty:
            continue
        headroom = float(plant_row["여유"].iloc[0])
        keys = grp["key"].unique().tolist()
        weights = {k: float(contract_by_key.get(k, 0.0)) for k in keys}
        positive = [w for w in weights.values() if w > 0]
        if positive:
            # 계약 자료에 없는 연결 지역은 같은 정수장의 다른 지역 평균 가중치를 준다(0으로 만들지 않음)
            fill = sum(positive) / len(positive)
            weights = {k: (w if w > 0 else fill) for k, w in weights.items()}
        else:
            weights = {k: 1.0 for k in keys}
        total_w = sum(weights.values())
        for k, w in weights.items():
            kw_alloc_rows.append({"key": k, "정수장": plant_name, "여유_배분": headroom * (w / total_w),
                                  "여유_전체": headroom, "연결사업자수": len(keys)})
    kw_alloc = pd.DataFrame(kw_alloc_rows)
    H_kw_by_key = kw_alloc.groupby("key")["여유_배분"].sum() if not kw_alloc.empty else pd.Series(dtype=float)
    H_kw_full_by_key = kw_alloc.groupby("key")["여유_전체"].sum() if not kw_alloc.empty else pd.Series(dtype=float)
    sharers_by_key = kw_alloc.groupby("key")["연결사업자수"].max() if not kw_alloc.empty else pd.Series(dtype=float)


    # (정수수입)X: 다른 지자체 소유 정수장 X의 정수를 받아 쓰는 연결(예: 계룡·세종 <- 대전 월평·신탄진).
    # 수입 계약량 자료가 없어 정수장 X의 여유를 소유 지자체와 수입 지자체들이 균등하게 나눈다.
    # 같은 물을 두 번 세지 않도록 수입 지자체 몫만큼 소유 지자체의 자체 여유에서 뺀다.
    import_links = detail[detail["시설명"].astype(str).str.startswith("(정수수입)")].copy()
    import_links["원정수장"] = import_links["시설명"].str.replace("(정수수입)", "", regex=False)
    import_share: dict[str, float] = {}
    import_full: dict[str, float] = {}
    owner_deduct: dict[str, float] = {}
    for plant_name, grp in import_links.groupby("원정수장"):
        src = own_rows[own_rows["시설명"] == plant_name]
        if src.empty:
            continue
        owner = src["key"].iloc[0]
        headroom = max(float((src["설계시설용량"] - src["일최대생산"]).sum()), 0.0)
        importers = [k for k in grp["key"].unique() if k != owner]
        if not importers:
            continue
        share = headroom / (len(importers) + 1)
        for k in importers:
            import_share[k] = import_share.get(k, 0.0) + share
            import_full[k] = import_full.get(k, 0.0) + headroom
        owner_deduct[owner] = owner_deduct.get(owner, 0.0) + share * len(importers)

    own_by_key = own_rows.groupby("key").agg(
        H_own_T=("H_own_T", "sum"), 설계시설용량_합=("설계시설용량", "sum"), 일최대생산_합=("일최대생산", "sum")
    ).reset_index()
    own_by_key["가동률_최대"] = np.where(
        own_by_key["설계시설용량_합"] > 0, own_by_key["일최대생산_합"] / own_by_key["설계시설용량_합"] * 100, np.nan
    )
    intake_by_key = intake_own.groupby("key").agg(H_own_I=("H_own_I", "sum")).reset_index()

    regions = regions.merge(own_by_key, on="key", how="left")
    regions = regions.merge(intake_by_key, on="key", how="left")
    # 기본값(H_kw, H_imp, H_own)은 연결된 정수장 여유를 지자체끼리 나누지 않는다. 연결된 정수장의 여유는
    # 특정 지자체 몫이 아닌 공용 여유이므로, 한 곳이 추가 공급을 받는다면 최대 그만큼 가능하다는 상한이다.
    # 같은 정수장에 연결된 지역끼리 같은 여유를 공유하므로 동시에 쓸 수 있는 양이 아니다(공유정수장_사업자수 참고).
    # *_배분 열은 이전 방식(공급 비중 배분, 수입분은 소유·수입 지자체가 균등 분할)의 참고값이다.
    regions["H_kw"] = regions["key"].map(H_kw_full_by_key).fillna(0.0)
    regions["H_imp"] = regions["key"].map(import_full).fillna(0.0)
    regions["H_kw_배분"] = regions["key"].map(H_kw_by_key).fillna(0.0)
    regions["H_imp_배분"] = regions["key"].map(import_share).fillna(0.0)
    regions["공유정수장_사업자수"] = regions["key"].map(sharers_by_key).fillna(0).astype(int)
    regions["운휴정수장_용량"] = regions["key"].map(idle_by_key).fillna(0.0)
    regions["원수수입"] = regions["key"].map(raw_import).fillna(0) > 0

    def _own(t):
        def f(r):
            if pd.notna(r[t]) and pd.notna(r["H_own_I"]) and not r["원수수입"]:
                return min(r[t], r["H_own_I"])
            return r[t] if pd.notna(r[t]) else np.nan
        return f

    regions["H_own_T_배분"] = regions["H_own_T"] - regions["key"].map(owner_deduct).fillna(0.0)
    regions["H_own"] = regions.apply(_own("H_own_T"), axis=1)
    regions["H_own_배분"] = regions.apply(_own("H_own_T_배분"), axis=1)

    # 광역정수장 연결 여부는 배분 여유(H_kw)가 아니라 연결행 존재로 판단한다. 연결된 광역정수장이
    # 과부하(예: 보령 102.5%)면 H_kw=0이지만 공급원은 있으므로 "미매칭"이 아니다.
    has_kw_link = regions["key"].isin(set(kw_link_rows["key"]))
    has_import = regions["key"].isin(set(import_share))
    parts = pd.DataFrame({
        "자체": regions["H_own"].notna(), "광역": has_kw_link, "정수수입": has_import,
    })
    regions["산출근거"] = parts.apply(lambda r: "+".join(c for c in parts.columns if r[c]) or "미매칭", axis=1)

    regions["H_own_clip"] = regions["H_own"].fillna(0.0).clip(lower=0)
    regions["H"] = regions["H_own_clip"] + regions["H_kw"] + regions["H_imp"]
    regions["H_배분"] = regions["H_own_배분"].fillna(0.0).clip(lower=0) + regions["H_kw_배분"] + regions["H_imp_배분"]
    regions["H_자체만"] = regions["H_own_clip"]
    regions["광역의존비중"] = np.where(regions["H"] > 0, (regions["H_kw"] + regions["H_imp"]) / regions["H"], np.nan)

    # 자체 정수시설 설계용량이 0(=시설 자체가 없어 광역에 전적으로 의존)이면 가동률 0.0은
    # "여유롭다"가 아니라 "측정 불가"이므로 결측 처리 -> S_stress에서 중립값(0.5)으로 처리됨
    has_own_facility = regions["설계시설용량_합"].fillna(0) > 0
    regions["U"] = np.where(has_own_facility, regions["가동률_최대"] / 100.0, np.nan)

    # 물 수요 기준값 D = P_IT x 24h x pWUE (근거: datacenter-dataset/기준값D_근거.md).
    # 기본: 1GW x 24h x 0.17 L/kWh = 4,080 m3/일(공랭 WUE, 첨두계수 미적용). 근거: 기준값D_근거.md
    # 민감도: 정부 구두보고 400, 수랭 WUE 1.41 -> 33,840.
    # H_미절단: H_own의 음수(자체 정수장 과부하 = 부족분)를 자르지 않고 합산한 값(참고용).
    regions["H_미절단"] = regions["H_own"].fillna(0.0) + regions["H_kw"] + regions["H_imp"]

    regions["S_stress_추정"] = regions["U"].isna()
    regions["S_stress"] = ((1.0 - regions["U"]) / 0.3).clip(lower=0, upper=1)
    regions["S_stress"] = regions["S_stress"].fillna(0.5)

    regions["D_기준"] = D_BASE
    C = (regions["H"] / D_BASE).clip(upper=5)
    regions["S_cov"] = np.log1p(C) / np.log1p(5)
    regions["탈락여부"] = regions["H"] < D_BASE
    regions["탈락여부_자체만"] = regions["H_자체만"] < D_BASE
    # H(연결 정수장 여유 전체, 상한)로는 통과하지만 자체 정수장만으로는 탈락: 광역 공급 협의에 달린 지역
    regions["광역협의필요"] = ~regions["탈락여부"] & regions["탈락여부_자체만"]
    regions["Water_final"] = np.where(regions["탈락여부"], 0.0, 0.7 * regions["S_cov"] + 0.3 * regions["S_stress"])
    for label, D in D_SENSITIVITY.items():
        regions[f"탈락여부_{label}"] = regions["H"] < D

    # 상위 시 값 공유: 상수도 사업자가 시 단위로 하나라 구·읍면동별 값이 따로 없는 행(광역시 구·군,
    # 세종 읍면동, 청주·포항의 구). 같은 key를 여러 후보 행이 나눠 쓰면 True.
    key_counts = regions["key"].map(regions["key"].value_counts())
    regions["상위시값공유"] = key_counts > 1

    projects = regions.apply(lambda r: expansion_projects_for(r["시도"], r["시군구"]), axis=1)
    regions["확충사업_대상"] = projects.map(bool)
    regions["확충사업명"] = projects.map(lambda p: ", ".join(p))

    industrial = load_industrial_water_by_key(xlsx_path, token_index)
    regions = regions.merge(industrial, on="key", how="left")
    regions["공업용수_여유"] = regions["공업용수_여유"].fillna(0.0)
    regions["공업용수_시설"] = regions["공업용수_시설"].fillna("")

    regions["기준연도"] = 2024  # 2024년 상수도통계 기준
    out_cols = [
        "시군구코드", "개편전코드", "시도", "시군구", "기준연도", "key", "상위시값공유", "산출근거", "원수수입",
        "H_own", "H_kw", "H_imp", "H", "H_자체만", "H_배분", "H_kw_배분", "H_imp_배분", "광역의존비중",
        "공유정수장_사업자수", "H_미절단", "U", "S_stress", "S_stress_추정",
        "D_기준", "S_cov", "Water_final", "탈락여부", "탈락여부_자체만", "광역협의필요",
        *[f"탈락여부_{k}" for k in D_SENSITIVITY],
        "확충사업_대상", "확충사업명", "공업용수_여유", "공업용수_시설", "운휴정수장_용량",
    ]
    result = regions[out_cols]

    result.to_csv(out_dir / "수자원_피처.csv", index=False, encoding="utf-8-sig")
    kw_alloc.to_csv(out_dir / "광역배분_상세.csv", index=False, encoding="utf-8-sig")

    balance["key"] = balance["사업자원문"].map(lambda s: key_from_raw_name(s) if is_detail_utility_row(s) else None)
    balance_check = balance.dropna(subset=["key"]).merge(
        result[["key", "H_kw"]], on="key", how="right"
    )
    balance_check.to_csv(out_dir / "검증_광역수입량_비교.csv", index=False, encoding="utf-8-sig")

    print("산출근거 분포:")
    print(result["산출근거"].value_counts().to_string())
    print()
    mism = result[result["산출근거"] == "미매칭"][["시도", "시군구"]]
    print(f"미매칭 지역 ({len(mism)}개):")
    print(mism.to_string())
    print()
    check_regions = ["태안", "횡성", "음성", "영양", "울릉", "고령", "청양", "아산", "나주", "칠곡", "증평", "전주", "군산", "춘천", "고성", "광주시"]
    print("검증 대상 지역:")
    print(result[result["시군구"].str.contains("|".join(check_regions))][
        ["시도", "시군구", "산출근거", "H_own", "H_kw", "H", "U"]
    ].to_string())

    return result


if __name__ == "__main__":
    if not XLSX_PATH.exists():
        print(f"xlsx 파일 없음: {XLSX_PATH}", file=sys.stderr)
        sys.exit(1)
    if not REGION_LIST_PATH.exists():
        print(f"기준 행 목록 없음: {REGION_LIST_PATH}", file=sys.stderr)
        sys.exit(1)
    build(XLSX_PATH, REGION_LIST_PATH, OUT_DIR)
