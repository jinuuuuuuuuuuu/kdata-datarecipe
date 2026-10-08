# AWS(방재기상관측) 시간자료(awsh.php) 전국 일괄 수집: 2024-01-01 00:00 ~ 2025-12-31 23:00 (KST)
# ASOS(collect_hourly.py)로 커버 안 되는 144개 후보 시군구 중 49개를 채우기 위한 보조 수집.
# disp=1이면 JSON으로 응답하므로 파싱이 ASOS(고정폭 텍스트)보다 간단하다.
# ASOS와 동일하게 stn 파라미터를 비우면(또는 생략하면) 전 지점을 한 번에 반환한다.
import http.client
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"
START = datetime(2024, 1, 1, 0, 0)
END = datetime(2025, 12, 31, 23, 0)
OUT_DIR = ROOT / "data" / "raw" / "cooling" / "aws"
PROGRESS_FILE = ROOT / "data" / "raw" / "cooling" / "_log" / "progress_aws.txt"
RETRY = 5
SLEEP_BETWEEN_CALLS = 0.05
SLEEP_ON_ERROR = 3
FAILED_LOG = ROOT / "data" / "raw" / "cooling" / "_log" / "failed_hours_aws.txt"
PARALLEL_WORKERS = 3
BATCH_HOURS = 48

COLUMNS = ["TM", "AWS_ID", "TA", "WD", "WS", "RN_DAY", "RN_HR1", "HM", "PA", "PS"]


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


def fetch_hour(tm: datetime, auth_key: str) -> list[dict]:
    tm_str = tm.strftime("%Y%m%d%H%M")
    url = f"{BASE_URL}?tm={tm_str}&disp=1&authKey={auth_key}"
    last_err = None
    req = Request(url, headers={"User-Agent": "curl/8.0"})
    for attempt in range(1, RETRY + 1):
        try:
            with urlopen(req, timeout=8) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError, http.client.IncompleteRead) as e:
            last_err = e
            time.sleep(SLEEP_ON_ERROR * attempt)
    raise RuntimeError(f"{tm_str} 수집 실패 ({RETRY}회 재시도): {last_err}")


def main() -> None:
    import csv

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
    print(f"AWS 수집 시작: {cur} (전체 {total_hours}시간 중 {done_hours}시간 완료 상태)")

    from concurrent.futures import ThreadPoolExecutor

    month_key = None
    csv_file = None
    writer = None

    # 일부 요청이 응답 없이 제한시간까지 매달리는 현상이 잦아, BATCH_HOURS 단위로 동시에 요청한 뒤
    # 시간 순서대로 기록한다(진행 파일은 배치 안에서 한 시간씩 갱신되므로 이어받기 방식은 동일).
    def safe_fetch(tm: datetime):
        try:
            return tm, fetch_hour(tm, auth_key), None
        except RuntimeError as e:
            return tm, None, e

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
            while cur <= END:
                batch = []
                t = cur
                while t <= END and len(batch) < BATCH_HOURS:
                    batch.append(t)
                    t += timedelta(hours=1)
                results = list(pool.map(safe_fetch, batch))

                for tm, rows, err in results:
                    key = tm.strftime("%Y%m")
                    if key != month_key:
                        if csv_file:
                            csv_file.close()
                        month_key = key
                        out_path = OUT_DIR / f"aws_hourly_{key}.csv"
                        is_new = not out_path.exists()
                        csv_file = open(out_path, "a", newline="", encoding="utf-8")
                        writer = csv.writer(csv_file)
                        if is_new:
                            writer.writerow(COLUMNS)

                    if err is None:
                        for row in rows:
                            writer.writerow([row.get(c, "") for c in COLUMNS])
                        csv_file.flush()
                    else:
                        print(f"[SKIP] {tm} 수집 실패, 건너뜀: {err}")
                        with open(FAILED_LOG, "a", encoding="utf-8") as f:
                            f.write(tm.strftime("%Y%m%d%H%M") + "\n")

                    save_progress(tm)
                    done_hours += 1
                    if done_hours % 100 == 0:
                        pct = done_hours / total_hours * 100
                        print(f"[{done_hours}/{total_hours}] {pct:.1f}% - {tm} 완료", flush=True)

                cur = t

    finally:
        if csv_file:
            csv_file.close()

    print("AWS 전체 수집 완료.")


if __name__ == "__main__":
    main()
