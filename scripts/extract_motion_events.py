import json
import sys
from pathlib import Path
import glob

BASE_DIR = Path(__file__).resolve().parent.parent

# 최신 pose_data 파일 자동 찾기
pose_files = glob.glob(str(BASE_DIR / "outputs" / "pose_data_*.json"))
if pose_files:
    INPUT_JSON = Path(max(pose_files, key=lambda x: Path(x).stat().st_mtime))
    print(f"Found {len(pose_files)} pose files, using latest: {INPUT_JSON.name}")
else:
    INPUT_JSON = BASE_DIR / "outputs" / "pose_data.json"
    print(f"No pose_data_*.json files found, using default: {INPUT_JSON.name}")

OUTPUT_JSON = BASE_DIR / "outputs" / "motion_events.json"

# ===== 설정값 =====

THRESHOLD = 0.05

FPS = 30

GRACE_FRAMES = int(0.7 * FPS)
MIN_DURATION = 1.0
MIN_PEAK_SCORE = 0.15

# ==================

def extract_motion_events(input_json: Path, output_json: Path) -> int:
    """
    Pose data에서 motion event 추출
    
    Args:
        input_json: 입력 pose_data JSON 파일
        output_json: 출력 motion_events JSON 파일
    
    Returns:
        추출된 이벤트 개수
    """
    input_json = Path(input_json)
    output_json = Path(output_json)
    
    # 입력 파일 확인
    if not input_json.exists():
        print(f"✗ Input file not found: {input_json}")
        raise FileNotFoundError(f"Input file not found: {input_json}")
    
    # 출력 디렉토리 생성
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON 로드
    print(f"[1/3] Loading pose data from {input_json.name}...")
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            pose_data = json.load(f)
        print(f"  ✓ Loaded {len(pose_data)} frames")
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        raise

    # Motion event 추출
    print(f"[2/3] Extracting motion events...")
    print(f"  Config: threshold={THRESHOLD}, grace_frames={GRACE_FRAMES}, min_duration={MIN_DURATION}s")
    
    # 디버깅: threshold를 넘는 프레임 수 확인
    high_motion_frames = sum(1 for frame in pose_data if frame["movement_score"] > THRESHOLD)
    print(f"  Debug: {high_motion_frames} frames exceed threshold {THRESHOLD}")
    
    events = []
    in_event = False
    event_start = None
    event_scores = []
    event_tags = set()
    low_motion_count = 0

    for frame in pose_data:

        timestamp = frame["timestamp"]
        score = frame["movement_score"]
        frame_tags = frame.get("gesture_tags", [])

        # 움직임 감지
        if score > THRESHOLD:

            if not in_event:
                in_event = True
                event_start = timestamp
                event_scores = []
                event_tags = set()

            event_scores.append(score)
            low_motion_count = 0
            if frame_tags:
                event_tags.update(frame_tags)

        # threshold 이하
        else:

            if in_event:
                if frame_tags:
                    event_tags.update(frame_tags)

                low_motion_count += 1

                # 아직 grace period 안 지남
                if low_motion_count <= GRACE_FRAMES:
                    event_scores.append(score)

                # grace period 초과 -> 이벤트 종료
                else:

                    event_end = timestamp
                    duration = event_end - event_start

                    avg_score = sum(event_scores) / len(event_scores)
                    peak_score = max(event_scores)

                    # 필터링: 너무 짧거나 약한 이벤트 제거
                    if (
                        duration >= MIN_DURATION
                        and peak_score >= MIN_PEAK_SCORE
                    ):

                        event_data = {
                            "start": round(event_start, 2),
                            "end": round(event_end, 2),
                            "duration": round(duration, 2),
                            "action": "high_motion",
                            "average_score": round(avg_score, 4),
                            "peak_score": round(peak_score, 4)
                        }
                        if event_tags:
                            event_data["gesture_tags"] = sorted(event_tags)
                        events.append(event_data)

                    in_event = False
                    low_motion_count = 0

    # 마지막 이벤트 처리
    if in_event:

        event_end = pose_data[-1]["timestamp"]
        duration = event_end - event_start

        avg_score = sum(event_scores) / len(event_scores)
        peak_score = max(event_scores)

        if (
            duration >= MIN_DURATION
            and peak_score >= MIN_PEAK_SCORE
        ):
            event_data = {
                "start": round(event_start, 2),
                "end": round(event_end, 2),
                "duration": round(duration, 2),
                "action": "high_motion",
                "average_score": round(avg_score, 4),
                "peak_score": round(peak_score, 4)
            }
            if event_tags:
                event_data["gesture_tags"] = sorted(event_tags)
            events.append(event_data)

    print(f"  ✓ Extracted {len(events)} motion events")

    # 결과 저장
    print(f"[3/3] Saving results...")
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        print(f"  ✓ Saved to: {output_json}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        raise
    
    return len(events)


if __name__ == "__main__":
    try:
        count = extract_motion_events(INPUT_JSON, OUTPUT_JSON)
        print(f"\n✓ Motion event extraction complete")
        print(f"  Total events: {count}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        sys.exit(1)
