# 물 수요 기준값(D)이 정해지면 탈락여부와 수자원 점수를 계산한다. 데이터셋에는 D에 의존하는 열이 없다.
# 사용: python apply_water_gate.py --mw 100 --wue 0.9 [--hours 24] [--in 수자원_냉각_통합.csv] [--out 결과.csv]
#       또는 python apply_water_gate.py --d 2160
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--mw", type=float, help="데이터센터 전력 규모(MW)")
ap.add_argument("--wue", type=float, help="물 사용 원단위(L/kWh)")
ap.add_argument("--hours", type=float, default=24, help="일 가동시간")
ap.add_argument("--d", type=float, help="물 수요(m3/일)를 직접 지정")
ap.add_argument("--h", default="H", choices=["H", "H_배분", "H_자체만"], help="비교에 쓸 공급 여유 열(기본 H=연결 정수장 여유 전체, H_자체만=보수적, H_배분=공급비중 배분)")
ap.add_argument("--in", dest="src", default=str(Path(__file__).resolve().parents[2] / "data" / "processed" / "수자원_냉각_통합.csv"))
ap.add_argument("--out", default="수자원_냉각_통합_D적용.csv")
a = ap.parse_args()

if a.d is not None:
    D = a.d
elif a.mw and a.wue:
    D = a.mw * 1000 * a.hours * a.wue / 1000  # kW x h x L/kWh / 1000 = m3/일
else:
    ap.error("--d 또는 --mw 와 --wue 를 지정하세요")

df = pd.read_csv(a.src, encoding="utf-8-sig", dtype={"시군구코드": str, "개편전코드": str})
pre = "수자원_" if "수자원_H" in df.columns else ""
H, S = df[pre + a.h], df[pre + "S_stress"]
cov = np.log1p((H / D).clip(upper=5)) / np.log1p(5)
df[pre + "D_m3일"] = D
df[pre + "D기준_H열"] = a.h
df[pre + "S_cov"] = cov
df[pre + "탈락여부"] = H < D
df[pre + "Water_final"] = np.where(H >= D, 0.7 * cov + 0.3 * S, 0.0)
df.to_csv(a.out, index=False, encoding="utf-8-sig")
print(f"D={D:g} m3/일 ({a.h}): 탈락 {int((H < D).sum())}행 / 통과 {int((H >= D).sum())}행 -> {a.out}")
