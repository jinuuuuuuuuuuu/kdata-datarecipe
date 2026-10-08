"""레시피 결과 이미지 5장 생성(recipe/figures/).

1 등급 지도(수도권 확대 포함)  2 수용 점수 지도  3 피처별 점수 지도(5개)
4 상위 20곳 점수 구성(가중 기여도)  5 확정 부지 검증과 가중치 민감도

경계: 통계청 SGIS 행정동 경계를 시군구로 합쳐 단순화한 파일(data/processed/boundary_sigungu_2026_simplified.geojson,
원본 vuski/admdongkor ver20260701, CC BY 4.0 / 공공누리 제1유형 출처표시).
"""
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "recipe" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 디자인 값(검증된 기본 팔레트) ----------
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # 파랑 순차(100/250/400/550/700)
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
GRADE_COLOR = {"A": "#184f95", "B": "#3987e5", "C": "#86b6ef"}  # 서열형: 한 색상, 진할수록 높은 등급(검증 통과)
WATER_COLOR = "#eb6834"  # 용수선결: 파랑 계열과 구분되는 주황
NEUTRAL_EXCL, NEUTRAL_HOLD = "#d4d3cd", "#8a8985"
FEAT_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # 범주형 슬롯 1~5(인접 쌍 검증 통과)
FEATS = [("power", "전력", 0.4175), ("renewable", "신재생", 0.1922), ("water", "수자원", 0.1639),
         ("disaster", "재해", 0.1297), ("cooling", "냉각", 0.0968)]

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
})
ASPECT = 1 / np.cos(np.deg2rad(36))
CREDIT = "경계: 통계청 SGIS 행정동 경계(admdongkor ver20260701, CC BY 4.0·공공누리 1유형) 시군구 합산"

d = pd.read_csv(ROOT / "recipe" / "aidc_siting_dataset.csv", encoding="utf-8-sig", dtype={"sigungu_code": str})
# 제출용 데이터셋에는 순위 열이 없으므로 상세 순위 파일에서 가져온다(그림 4·5에서 사용)
_rank = pd.read_csv(ROOT / "data" / "processed" / "최종_순위.csv", encoding="utf-8-sig", dtype={"시군구코드": str})
_rank = _rank[["시군구코드", "전체순위"]].rename(columns={"시군구코드": "sigungu_code", "전체순위": "national_rank"})
d = d.merge(_rank, on="sigungu_code", validate="1:1")
gj = gpd.read_file(ROOT / "data" / "processed" / "boundary_sigungu_2026_simplified.geojson")
gdf = gj.merge(d, on="sigungu_code", validate="1:1")
assert len(gdf) == 256


def style_map(ax, xlim=None, ylim=None):
    ax.set_aspect(ASPECT)
    ax.set_axis_off()
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def base(ax, lw=0.25):
    gdf.plot(ax=ax, facecolor="none", edgecolor=SURFACE, linewidth=lw)


def title(fig, main, sub):
    gap = 0.46 / fig.get_figheight()  # 제목과 부제 사이 간격을 그림 높이와 무관하게 약 0.46인치로 유지
    fig.text(0.04, 0.965, main, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.965 - gap, sub, fontsize=10.5, color=INK2, va="top")


def credit(fig, extra=""):
    fig.text(0.04, 0.012, (extra + "  " if extra else "") + CREDIT, fontsize=7, color=INK3, va="bottom")


# ================= 1. 등급 지도 =================
def draw_grades(ax, lw):
    for g, c in GRADE_COLOR.items():
        gdf[gdf.suitability_grade == g].plot(ax=ax, facecolor=c, edgecolor=SURFACE, linewidth=lw)
    gdf[gdf.suitability_grade == "용수선결"].plot(ax=ax, facecolor=WATER_COLOR, edgecolor=SURFACE, linewidth=lw)
    gdf[gdf.suitability_grade == "제외"].plot(ax=ax, facecolor=NEUTRAL_EXCL, edgecolor=SURFACE, linewidth=lw)
    h = gdf[gdf.suitability_grade == "판정보류"]  # 피처 결측으로 등급을 보류한 지역(현재 데이터에는 없음)
    if len(h):
        h.plot(ax=ax, facecolor=NEUTRAL_EXCL, edgecolor=NEUTRAL_HOLD, linewidth=0.3, hatch="//////")


fig = plt.figure(figsize=(9, 10.2))
ax = fig.add_axes([0.02, 0.07, 0.96, 0.80])
draw_grades(ax, 0.2)
style_map(ax, (124.55, 132.1), (33.0, 38.7))
# 수도권 확대
bx, by = (126.55, 127.45), (37.0, 37.95)
ax.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0], fill=False, edgecolor=INK2, linewidth=0.8))
ins = fig.add_axes([0.68, 0.08, 0.30, 0.31])
draw_grades(ins, 0.3)
style_map(ins, bx, by)
for s in ins.spines.values():
    s.set_visible(False)
ins.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0], fill=False, edgecolor=INK2, linewidth=0.8))
ins.text(bx[0] + 0.01, by[1] - 0.02, "수도권 확대", fontsize=9, color=INK2, va="top")
n = d.suitability_grade.value_counts()
N_EVAL = int(d.suitability_score.notna().sum())  # 평가 대상(후보제외를 뺀 지역)
N_EXCL = int(d.suitability_score.isna().sum())
N_ZERO = int(((d.power_substation_headroom_mw_2029 == 0) & d.suitability_score.notna()).sum())  # 전력 여유 0MW
title(fig, "수용 등급 지도: 256개 시군구를 A·B·C·용수선결·제외로 구분",
      f"A {n['A']}곳 · B {n['B']}곳 · C {n['C']}곳 · 용수선결 {n['용수선결']}곳 · 제외 {n['제외']}곳")
handles = [Patch(facecolor=GRADE_COLOR["A"], label="A (용수 통과 지역의 상위 20%)"),
           Patch(facecolor=GRADE_COLOR["B"], label="B (20–50%)"),
           Patch(facecolor=GRADE_COLOR["C"], label="C (하위 50%)"),
           Patch(facecolor=WATER_COLOR, label="용수선결 (공급 여유 4,080 m³/일 미만)"),
           Patch(facecolor=NEUTRAL_EXCL, label="제외 (연안 고위험)")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.04, 0.04), frameon=False, fontsize=9, labelcolor=INK2,
           ncol=2, columnspacing=1.6, handlelength=1.4)
credit(fig)
fig.savefig(OUT / "fig1_grade_map.png", dpi=200)
plt.close(fig)

# ================= 2. 수용 점수 지도 =================
fig = plt.figure(figsize=(9, 10.2))
ax = fig.add_axes([0.02, 0.07, 0.96, 0.80])
vmin, vmax = d.suitability_score.min(), d.suitability_score.max()
norm = Normalize(vmin, vmax)
gdf[gdf.suitability_score.isna()].plot(ax=ax, facecolor=NEUTRAL_EXCL, edgecolor=SURFACE, linewidth=0.2)
gdf[gdf.suitability_score.notna()].plot(ax=ax, column="suitability_score", cmap=CMAP, norm=norm, edgecolor=SURFACE, linewidth=0.2)
style_map(ax, (124.55, 132.1), (33.0, 38.7))
ax.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0], fill=False, edgecolor=INK2, linewidth=0.8))
ins = fig.add_axes([0.68, 0.08, 0.30, 0.31])
gdf[gdf.suitability_score.isna()].plot(ax=ins, facecolor=NEUTRAL_EXCL, edgecolor=SURFACE, linewidth=0.3)
gdf[gdf.suitability_score.notna()].plot(ax=ins, column="suitability_score", cmap=CMAP, norm=norm, edgecolor=SURFACE, linewidth=0.3)
style_map(ins, bx, by)
ins.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0], fill=False, edgecolor=INK2, linewidth=0.8))
ins.text(bx[0] + 0.01, by[1] - 0.02, "수도권 확대", fontsize=9, color=INK2, va="top")
title(fig, "수용 점수 지도: 점수가 높을수록 1GW급 수용 여건이 좋음",
      f"전력·신재생·수자원·재해·냉각 5개 점수의 가중합(0–1), 실제 범위 {vmin:.2f}–{vmax:.2f}. 회색은 후보제외 {N_EXCL}곳")
cax = fig.add_axes([0.06, 0.055, 0.34, 0.014])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, length=0, colors=INK2)
fig.text(0.06, 0.082, "수용 점수", fontsize=9, color=INK2)
credit(fig)
fig.savefig(OUT / "fig2_score_map.png", dpi=200)
plt.close(fig)

# ================= 3. 피처별 점수 지도 =================
fig, axes = plt.subplots(2, 3, figsize=(14, 10.6))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.07, wspace=0.02, hspace=0.12)
norm01 = Normalize(0, 1)
for ax, (key, label, w) in zip(axes.ravel(), FEATS):
    col = f"{key}_score"
    gdf[gdf[col].isna()].plot(ax=ax, facecolor=NEUTRAL_EXCL, edgecolor=SURFACE, linewidth=0.1)
    gdf[gdf[col].notna()].plot(ax=ax, column=col, cmap=CMAP, norm=norm01, edgecolor=SURFACE, linewidth=0.1)
    style_map(ax, (124.55, 132.1), (33.0, 38.7))
    ax.set_title(f"{label} (가중치 {w * 100:.1f}%)", fontsize=12, color=INK, loc="left", fontweight="bold")
ax = axes.ravel()[5]
ax.axis("off")
cax = fig.add_axes([0.70, 0.30, 0.24, 0.014])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm01, cmap=CMAP), cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=9, length=0, colors=INK2)
fig.text(0.70, 0.335, "피처 점수(0–1, 클수록 유리)", fontsize=10, color=INK2)
fig.text(0.70, 0.245, f"회색: 후보제외\n(전력은 평가 대상의 약 {N_ZERO / N_EVAL:.0%}({N_ZERO}곳)가\n변전소 여유 0MW라 점수 0)",
         fontsize=9, color=INK2, va="top")
title(fig, "피처별 점수 지도: 같은 지역도 전력·신재생·수자원·재해·냉각 여건이 서로 다름",
      f"평가 대상 {N_EVAL}곳 안에서 피처마다 0–1로 정규화한 점수. 가중치는 논문 AHP 종합중요도를 5개 피처로 재정규화")
credit(fig)
fig.savefig(OUT / "fig3_feature_maps.png", dpi=200)
plt.close(fig)

# ================= 4. 상위 20곳 점수 구성 =================
top = d[d.national_rank.notna()].sort_values(["national_rank", "sigungu_code"]).head(20).copy()
SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "대전광역시": "대전", "울산광역시": "울산",
              "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
              "전북특별자치도": "전북", "전라남도": "전남", "전남광주통합특별시": "전남광주", "경상북도": "경북", "경상남도": "경남",
              "제주특별자치도": "제주"}
top["label"] = top.apply(lambda r: f"{int(r.national_rank)}. {SIDO_SHORT[r.sido_name]} {r.sigungu_name}"
                         + ("  [용수선결]" if r.suitability_grade == "용수선결" else ""), axis=1)
fig, ax = plt.subplots(figsize=(10.5, 8.2))
fig.subplots_adjust(left=0.27, right=0.95, top=0.84, bottom=0.11)
y = np.arange(len(top))[::-1]
left = np.zeros(len(top))
for (key, label, w), c in zip(FEATS, FEAT_COLOR):
    contrib = (top[f"{key}_score"].fillna(0) * w).values
    ax.barh(y, contrib, left=left, height=0.72, color=c, edgecolor=SURFACE, linewidth=1.2, label=f"{label} ({w * 100:.1f}%)")
    left += contrib
for yi, s in zip(y, top.suitability_score):
    ax.text(s + 0.008, yi, f"{s:.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(top.label, fontsize=9.5, color=INK)
ax.set_xlim(0, 0.9)
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("수용 점수(피처 점수 × 가중치의 합)", fontsize=9.5)
fig.legend(loc="upper left", bbox_to_anchor=(0.27, 0.885), ncol=5, frameon=False, fontsize=9, labelcolor=INK2,
           handlelength=1.2, columnspacing=1.2)
title(fig, "상위 20곳의 점수 구성: 전력 기여가 가장 크고, 나머지는 지역마다 조합이 다름",
      "막대 길이 = 수용 점수, 색 구간 = 피처별 기여(점수 × 가중치)")
credit(fig, "[용수선결]은 점수는 높지만 용수 기준 미달.")
fig.savefig(OUT / "fig4_top20_contribution.png", dpi=200)
plt.close(fig)

# ================= 5. 확정 부지 검증 + 민감도 =================
ranked = d[d.national_rank.notna()].sort_values("national_rank")
sites = {"세종특별자치시": ("세종", "세종"), "울주군": ("울산", "울산 울주군"), "동해시": ("강원", "동해시"), "울산광역시|남구": ("울산", "울산 남구")}
sel = {}
for _, r in ranked.iterrows():
    if r.sigungu_name == "세종특별자치시":
        sel["세종"] = r
    elif r.sido_name == "울산광역시" and r.sigungu_name == "울주군":
        sel["울산 울주군"] = r
    elif r.sigungu_name == "동해시":
        sel["강원 동해시"] = r
    elif r.sido_name == "울산광역시" and r.sigungu_name == "남구":
        sel["울산 남구"] = r
assert len(sel) == 4
fig = plt.figure(figsize=(13, 6.2))
axl = fig.add_axes([0.06, 0.12, 0.50, 0.68])
axr = fig.add_axes([0.70, 0.12, 0.27, 0.68])
axl.scatter(ranked.national_rank, ranked.suitability_score, s=14, color="#c7c6c0", linewidths=0)
OFFSET = {"세종": (14, 0.115), "울산 울주군": (26, 0.045), "강원 동해시": (14, 0.075), "울산 남구": (14, -0.075)}
for name, r in sel.items():
    dx, dy = OFFSET[name]
    axl.scatter([r.national_rank], [r.suitability_score], s=70, color="#184f95", edgecolor=SURFACE, linewidth=1.5, zorder=3)
    axl.annotate(f"{name} {int(r.national_rank)}위({r.suitability_grade})", (r.national_rank, r.suitability_score),
                 xytext=(r.national_rank + dx, r.suitability_score + dy), fontsize=9.5, color=INK, va="center",
                 arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.7, shrinkA=2, shrinkB=5))
axl.set_xlabel("전체 순위(1위가 가장 유리)", fontsize=9.5)
axl.set_ylabel("수용 점수", fontsize=9.5)
axl.yaxis.grid(True, color=GRID, linewidth=0.6)
axl.set_axisbelow(True)
for s in ("top", "right"):
    axl.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axl.spines[s].set_color(GRID)
axl.set_title("정부 확정 부지 4곳의 위치", loc="left", fontsize=11.5, fontweight="bold")

alts = [("Fuzzy-AHP 가중치", "민감도_순위_Fuzzy-AHP"), ("신재생 대체 배분", "민감도_순위_신재생대체배분"), ("동일 가중치", "민감도_순위_동일가중")]
rk = pd.read_csv(ROOT / "data" / "processed" / "최종_순위.csv", encoding="utf-8-sig", dtype={"시군구코드": str})
rows = []
for lab, col in alts:
    m = rk.전체순위.notna() & rk[col].notna()
    rho = rk.loc[m, "전체순위"].corr(rk.loc[m, col], method="spearman")
    t0 = set(rk[m].nsmallest(20, "전체순위").시군구코드)
    t1 = set(rk[m].nsmallest(20, col).시군구코드)
    rows.append((lab, rho, len(t0 & t1)))
yy = np.arange(len(rows))[::-1]
axr.barh(yy, [r[1] for r in rows], height=0.55, color="#2a78d6")
for yi, (lab, rho, ov) in zip(yy, rows):
    axr.text(rho + 0.015, yi, f"{rho:.3f}", va="center", fontsize=10, color=INK)
    axr.text(0.0, yi - 0.40, f"상위 20곳 중 {ov}곳 동일", fontsize=8.5, color=INK2, va="top")
axr.set_yticks(yy)
axr.set_yticklabels([r[0] for r in rows], fontsize=10)
axr.set_xlim(0, 1.15)
axr.set_ylim(-0.85, len(rows) - 0.4)
axr.xaxis.grid(True, color=GRID, linewidth=0.6)
axr.set_axisbelow(True)
axr.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    axr.spines[s].set_visible(False)
axr.spines["bottom"].set_color(GRID)
axr.set_xlabel("기본 순위와의 순위 상관(스피어만)", fontsize=9.5)
axr.set_title("가중치·배분 방식을 바꾸면 순위가 얼마나 유지되는가", loc="left", fontsize=11.5, fontweight="bold")
title(fig, "검증: 확정 부지의 낮은 순위는 전력 병목으로 설명되고, 순위는 전문가 가중치 변형에 안정적",
      "세종·울주는 상위, 동해(전력 여유 90MW)·울산 남구(40MW)는 하위권. 동일 가중치로 바꾸면 상위 20곳 중 12곳만 같음")
credit(fig, "정부 확정 부지: 세종·동해·울산(2026-07-13 발표).")
fig.savefig(OUT / "fig5_validation_sensitivity.png", dpi=200)
plt.close(fig)
print("saved:", sorted(p.name for p in OUT.glob("*.png")))
for lab, rho, ov in rows:
    print(lab, round(rho, 3), ov)
for k, r in sel.items():
    print(k, int(r.national_rank), r.suitability_grade, r.suitability_score)
