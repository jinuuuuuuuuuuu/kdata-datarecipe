# 수자원·냉각 데이터셋을 시군구코드(5자리 행정코드) 기준으로 1:1 병합한다.
# 이름("서구", "동구", "고성군")은 여러 시도에 겹치므로 코드로 합친다.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 저장소 루트
import pandas as pd

WATER = ROOT / "data" / "processed" / "water" / "수자원_피처.csv"
COOLING = ROOT / "data" / "processed" / "cooling" / "냉각_피처_시간기반.csv"
BASE = ROOT / "data" / "raw" / "reference" / "행정시군구_기준행.csv"
OUT = ROOT / "data" / "processed" / "수자원_냉각_통합.csv"
KEY = "시군구코드"
ID_COLS = [KEY, "개편전코드", "시도", "시군구"]

water = pd.read_csv(WATER, encoding="utf-8-sig", dtype={KEY: str, "개편전코드": str}).fillna({"개편전코드": ""})
cooling = pd.read_csv(COOLING, encoding="utf-8-sig", dtype={KEY: str, "개편전코드": str}).fillna({"개편전코드": ""})
base = pd.read_csv(BASE, encoding="utf-8-sig", dtype=str).fillna("")

for name, df in (("수자원", water), ("냉각", cooling)):
    dup = df[df.duplicated(KEY, keep=False)]
    if not dup.empty:
        raise ValueError(f"{name} 데이터셋에 {KEY} 중복 행이 있습니다:\n{dup[ID_COLS]}")
    missing, extra = set(base[KEY]) - set(df[KEY]), set(df[KEY]) - set(base[KEY])
    if missing or extra:
        raise ValueError(f"{name} 행이 기준 행과 다릅니다. 누락 {sorted(missing)}, 초과 {sorted(extra)}")

# 이름 컬럼이 두 파일에서 같은지 확인한 뒤 한쪽만 남긴다
chk = water[ID_COLS].merge(cooling[ID_COLS], on=KEY, suffixes=("_w", "_c"))
for c in ID_COLS[1:]:
    if (chk[f"{c}_w"] != chk[f"{c}_c"]).any():
        raise ValueError(f"두 데이터셋의 {c} 값이 다릅니다")

water = water.rename(columns={c: f"수자원_{c}" for c in water.columns if c not in ID_COLS})
cooling = cooling.drop(columns=ID_COLS[1:]).rename(columns={c: f"냉각_{c}" for c in cooling.columns if c != KEY})
merged = water.merge(cooling, on=KEY, how="inner", validate="one_to_one").sort_values(KEY)
merged.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"병합 완료: {merged.shape[0]}행 x {merged.shape[1]}열 -> {OUT}")
