# failed_hours_aws.txt에 기록된 누락 시간만 재수집해서 해당 월 CSV에 추가한다.
# collect_hourly_aws.py의 fetch_hour 로직을 재사용한다.
import csv
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트

from collect_hourly_aws import COLUMNS, OUT_DIR, fetch_hour, load_auth_key

FAILED_LOG = ROOT / "data" / "raw" / "cooling" / "_log" / "failed_hours_aws.txt"
STILL_FAILED_LOG = ROOT / "data" / "raw" / "cooling" / "_log" / "failed_hours_aws_retry2.txt"


def main() -> None:
    auth_key = load_auth_key()
    if not FAILED_LOG.exists():
        print("failed_hours_aws.txt가 없습니다.")
        return

    hours = [line.strip() for line in FAILED_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"재수집 대상: {len(hours)}시간")

    still_failed = []
    files_to_append: dict[str, list[list[str]]] = {}

    for h in hours:
        tm = datetime.strptime(h, "%Y%m%d%H%M")
        try:
            rows = fetch_hour(tm, auth_key)
            key = tm.strftime("%Y%m")
            files_to_append.setdefault(key, [])
            for row in rows:
                files_to_append[key].append([row.get(c, "") for c in COLUMNS])
            print(f"[OK] {h} 재수집 성공 ({len(rows)}개 지점)")
        except RuntimeError as e:
            print(f"[FAIL] {h} 재수집도 실패: {e}")
            still_failed.append(h)

    for key, rows in files_to_append.items():
        out_path = OUT_DIR / f"aws_hourly_{key}.csv"
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        print(f"{out_path.name}에 {len(rows)}행 추가")

    if still_failed:
        STILL_FAILED_LOG.write_text("\n".join(still_failed) + "\n", encoding="utf-8")
        print(f"여전히 실패: {len(still_failed)}건 -> {STILL_FAILED_LOG}")
    else:
        print("전부 재수집 성공.")


if __name__ == "__main__":
    main()
