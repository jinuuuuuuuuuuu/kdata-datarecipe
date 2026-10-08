# ASOS 시간자료(kma_sfctm2) 전국 일괄 수집: 2024-01-01 00:00 ~ 2025-12-31 23:00 (KST)
# stn=0 이면 해당 시각 전국 지점을 한 번에 반환하므로, 시간당 1회 호출로 전 지점을 수집한다.
import os
import sys
import time
import csv
import http.client
from datetime import datetime, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
START = datetime(2024, 1, 1, 0, 0)
END = datetime(2025, 12, 31, 23, 0)
OUT_DIR = ROOT / "data" / "raw" / "cooling" / "asos"
PROGRESS_FILE = ROOT / "data" / "raw" / "cooling" / "_log" / "progress.txt"
RETRY = 5
SLEEP_BETWEEN_CALLS = 0.05
SLEEP_ON_ERROR = 3
FAILED_LOG = ROOT / "data" / "raw" / "cooling" / "_log" / "failed_hours.txt"

COLUMNS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP",
    "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
]


def load_auth_key() -> str:
    env_path = Path(__file__).parent / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("KMA_AUTH_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(".env에서 KMA_AUTH_KEY를 찾지 못했습니다")


def load_progress() -> datetime | None:
    if PROGRESS_FILE.exists():
        text = PROGRESS_FILE.read_text(encoding="utf-8").strip()
        if text:
            return datetime.strptime(text, "%Y%m%d%H%M")
    return None


def save_progress(tm: datetime) -> None:
    PROGRESS_FILE.write_text(tm.strftime("%Y%m%d%H%M"), encoding="utf-8")


def fetch_hour(tm: datetime, auth_key: str) -> str:
    tm_str = tm.strftime("%Y%m%d%H%M")
    url = f"{BASE_URL}?tm={tm_str}&stn=0&authKey={auth_key}"
    last_err = None
    req = Request(url, headers={"User-Agent": "curl/8.0"})
    for attempt in range(1, RETRY + 1):
        try:
            with urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (URLError, HTTPError, TimeoutError, OSError, http.client.IncompleteRead) as e:
            last_err = e
            time.sleep(SLEEP_ON_ERROR * attempt)
    raise RuntimeError(f"{tm_str} 수집 실패 ({RETRY}회 재시도): {last_err}")


def parse_lines(raw: str) -> list[list[str]]:
    rows = []
    for line in raw.splitlines():
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < len(COLUMNS):
            continue
        rows.append(parts[: len(COLUMNS)])
    return rows


def main() -> None:
    auth_key = load_auth_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    resume_from = load_progress()
    cur = resume_from + timedelta(hours=1) if resume_from else START
    if cur > END:
        print("이미 전체 구간 수집 완료.")
        return

    total_hours = int((END - START).total_seconds() // 3600) + 1
    done_hours = int((cur - START).total_seconds() // 3600)
    print(f"수집 시작: {cur} (전체 {total_hours}시간 중 {done_hours}시간 완료 상태)")

    month_key = None
    csv_file = None
    writer = None

    try:
        while cur <= END:
            key = cur.strftime("%Y%m")
            if key != month_key:
                if csv_file:
                    csv_file.close()
                month_key = key
                out_path = OUT_DIR / f"asos_hourly_{key}.csv"
                is_new = not out_path.exists()
                csv_file = open(out_path, "a", newline="", encoding="utf-8")
                writer = csv.writer(csv_file)
                if is_new:
                    writer.writerow(COLUMNS)

            try:
                raw = fetch_hour(cur, auth_key)
                rows = parse_lines(raw)
                for row in rows:
                    writer.writerow(row)
                csv_file.flush()
            except RuntimeError as e:
                # 한 시간 호출이 재시도까지 다 실패해도 전체를 죽이지 않고 건너뛴다.
                # 실패한 시각은 별도 로그에 남겨 나중에 재수집할 수 있게 한다.
                print(f"[SKIP] {cur} 수집 실패, 건너뜀: {e}")
                with open(FAILED_LOG, "a", encoding="utf-8") as f:
                    f.write(cur.strftime("%Y%m%d%H%M") + "\n")

            save_progress(cur)

            done_hours += 1
            if done_hours % 100 == 0:
                pct = done_hours / total_hours * 100
                print(f"[{done_hours}/{total_hours}] {pct:.1f}% - {cur} 완료")

            cur += timedelta(hours=1)
            time.sleep(SLEEP_BETWEEN_CALLS)

    finally:
        if csv_file:
            csv_file.close()

    print("전체 수집 완료.")


if __name__ == "__main__":
    main()
