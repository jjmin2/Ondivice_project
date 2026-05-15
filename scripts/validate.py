"""
System Validation & Testing Script
온디바이스 편집포인트 추출 시스템 검증 및 테스트
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def check_file(path: Path, description: str) -> bool:
    """파일 존재 여부 확인"""
    path = Path(path)
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"  ✓ {description}: {path.name} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  ✗ {description}: NOT FOUND")
        return False


def validate_json_structure(path: Path, expected_fields: List[str]) -> bool:
    """JSON 파일 구조 검증"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"    ✗ Root is not a list")
            return False
        
        if not data:
            print(f"    ⚠ Empty list")
            return True
        
        first_item = data[0]
        missing = [f for f in expected_fields if f not in first_item]
        
        if missing:
            print(f"    ✗ Missing fields: {missing}")
            return False
        
        print(f"    ✓ Structure valid ({len(data)} items)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"    ✗ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def analyze_motion_events(path: Path) -> Dict[str, Any]:
    """Motion events 분석"""
    print("\n  [Motion Events Analysis]")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            events = json.load(f)
        
        if not events:
            print("    ✗ No events found")
            return {}
        
        durations = [e["duration"] for e in events]
        peak_scores = [e["peak_score"] for e in events]
        avg_scores = [e["average_score"] for e in events]
        
        stats = {
            "total_events": len(events),
            "total_duration": sum(durations),
            "avg_event_duration": sum(durations) / len(durations),
            "min_event_duration": min(durations),
            "max_event_duration": max(durations),
            "avg_peak_score": sum(peak_scores) / len(peak_scores),
            "min_peak_score": min(peak_scores),
            "max_peak_score": max(peak_scores),
        }
        
        print(f"    Total events: {stats['total_events']}")
        print(f"    Total duration: {stats['total_duration']:.2f}s")
        print(f"    Avg event duration: {stats['avg_event_duration']:.2f}s")
        print(f"    Peak score range: [{stats['min_peak_score']:.3f} ~ {stats['max_peak_score']:.3f}]")
        
        return stats
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {}


def analyze_transcription(path: Path) -> Dict[str, Any]:
    """Transcription segments 분석"""
    print("\n  [Transcription Analysis]")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        
        if not segments:
            print("    ✗ No segments found")
            return {}
        
        durations = [s["end"] - s["start"] for s in segments]
        total_text = sum(len(s["text"]) for s in segments)
        
        stats = {
            "total_segments": len(segments),
            "total_duration": segments[-1]["end"],
            "total_characters": total_text,
            "avg_segment_duration": sum(durations) / len(durations),
            "min_segment_duration": min(durations),
            "max_segment_duration": max(durations),
        }
        
        print(f"    Total segments: {stats['total_segments']}")
        print(f"    Total duration: {stats['total_duration']:.2f}s")
        print(f"    Total text: {stats['total_characters']} characters")
        print(f"    Avg segment duration: {stats['avg_segment_duration']:.2f}s")
        
        return stats
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {}


def analyze_highlights(path: Path) -> Dict[str, Any]:
    """Highlight segments 분석"""
    print("\n  [Highlights Analysis]")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            highlights = json.load(f)
        
        if not highlights:
            print("    ✗ No highlights found")
            return {}
        
        scores = [h["highlight_score"] for h in highlights]
        event_types = {}
        for h in highlights:
            et = h.get("event_type", "unknown")
            event_types[et] = event_types.get(et, 0) + 1
        
        stats = {
            "total_highlights": len(highlights),
            "avg_highlight_score": sum(scores) / len(scores),
            "min_highlight_score": min(scores),
            "max_highlight_score": max(scores),
            "event_types": event_types,
        }
        
        print(f"    Total highlights: {stats['total_highlights']}")
        print(f"    Highlight score range: [{stats['min_highlight_score']:.3f} ~ {stats['max_highlight_score']:.3f}]")
        print(f"    Avg score: {stats['avg_highlight_score']:.3f}")
        print(f"    Event types: {event_types}")
        
        # Top 3 highlights
        print("\n    Top 3 Highlights:")
        for i, h in enumerate(highlights[:3], 1):
            print(f"      {i}. [{h['start']:.1f}s ~ {h['end']:.1f}s] "
                  f"(score: {h['highlight_score']:.3f}, type: {h.get('event_type', '?')})")
            print(f"         \"{h['text'][:60]}...\"")
        
        return stats
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {}


def check_system_state():
    """전체 시스템 상태 확인"""
    print("\n" + "=" * 70)
    print("  OnDevice Video Edit Point Extraction System - Status Check")
    print("=" * 70)
    
    # 1. 기본 파일 확인
    print("\n[1] Basic Files")
    pose_ok = check_file(
        OUTPUTS_DIR / "pose_data.json",
        "Pose data"
    )
    
    # 2. Motion events
    print("\n[2] Motion Events")
    motion_file = OUTPUTS_DIR / "motion_events.json"
    motion_ok = check_file(motion_file, "Motion events")
    
    motion_stats = {}
    if motion_ok:
        if not validate_json_structure(
            motion_file,
            ["start", "end", "duration", "action", "average_score", "peak_score"]
        ):
            motion_ok = False
        else:
            motion_stats = analyze_motion_events(motion_file)
    
    # 3. Transcription
    print("\n[3] Transcription")
    trans_file = OUTPUTS_DIR / "transcription_segments.json"
    trans_ok = check_file(trans_file, "Transcription")
    
    trans_stats = {}
    if trans_ok:
        if not validate_json_structure(
            trans_file,
            ["start", "end", "text", "language"]
        ):
            trans_ok = False
        else:
            trans_stats = analyze_transcription(trans_file)
    
    # 4. Highlights
    print("\n[4] Final Highlights")
    highlight_file = OUTPUTS_DIR / "highlight_segments.json"
    highlight_ok = check_file(highlight_file, "Highlights")
    
    highlight_stats = {}
    if highlight_ok:
        if not validate_json_structure(
            highlight_file,
            ["start", "end", "duration", "text", "highlight_score", "event_type"]
        ):
            highlight_ok = False
        else:
            highlight_stats = analyze_highlights(highlight_file)
    
    # 5. 시스템 준비도
    print("\n[5] System Readiness")
    
    readiness_levels = {
        "Motion extraction": "✓" if motion_ok else "✗",
        "Speech transcription": "✓" if trans_ok else "✗",
        "Highlight generation": "✓" if highlight_ok else "✗",
    }
    
    for item, status in readiness_levels.items():
        print(f"  {status} {item}")
    
    all_ready = motion_ok and trans_ok and highlight_ok
    
    # 6. 권장사항
    print("\n[6] Recommendations")
    
    if not motion_ok:
        print("  • Generate motion events: python extract_motion_events.py")
    
    if not trans_ok:
        print("  • Transcribe audio: python transcribe_audio.py --input <video.mp4>")
    
    if not highlight_ok and motion_ok and trans_ok:
        print("  • Extract highlights: python merge_whisper_pose.py")
    
    if all_ready:
        print("  ✓ System is ready for video editing workflow!")
        print("  • Export highlights to your video editor")
        print("  • Use for chapter markers or clip generation")
        print("  • Optimize editing efficiency with automatic detection")
    
    return all_ready


def create_sample_data():
    """샘플 데이터 생성 (테스트용)"""
    print("\n" + "=" * 70)
    print("  Creating Sample Data for Testing")
    print("=" * 70)
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Sample motion events
    motion_events = [
        {
            "start": 5.0,
            "end": 8.5,
            "duration": 3.5,
            "action": "high_motion",
            "average_score": 0.25,
            "peak_score": 0.42
        },
        {
            "start": 15.2,
            "end": 18.0,
            "duration": 2.8,
            "action": "high_motion",
            "average_score": 0.22,
            "peak_score": 0.38
        },
        {
            "start": 32.5,
            "end": 36.2,
            "duration": 3.7,
            "action": "high_motion",
            "average_score": 0.28,
            "peak_score": 0.51
        }
    ]
    
    motion_file = OUTPUTS_DIR / "motion_events.json"
    with open(motion_file, "w", encoding="utf-8") as f:
        json.dump(motion_events, f, indent=2)
    print(f"  ✓ Created sample motion_events.json ({len(motion_events)} events)")
    
    # 2. Sample transcription
    transcription = [
        {
            "start": 4.5,
            "end": 9.2,
            "text": "이 부분이 매우 중요합니다.",
            "language": "ko"
        },
        {
            "start": 14.8,
            "end": 19.5,
            "text": "핵심 포인트를 설명드리겠습니다.",
            "language": "ko"
        },
        {
            "start": 31.2,
            "end": 37.8,
            "text": "결론적으로 이것이 가장 중요한 결과입니다.",
            "language": "ko"
        }
    ]
    
    trans_file = OUTPUTS_DIR / "transcription_segments.json"
    with open(trans_file, "w", encoding="utf-8") as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Created sample transcription_segments.json ({len(transcription)} segments)")
    
    print("\n  Sample data created for testing pipeline")
    print("  Run: python merge_whisper_pose.py (to generate highlights)")


# ==============================
# 메인
# ==============================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="System Validation & Testing")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample data for testing")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    # System state check
    ready = check_system_state()
    
    print("\n" + "=" * 70 + "\n")
    
    if ready:
        print("✓ System Status: READY")
        print("  All required data files are present and valid")
        sys.exit(0)
    else:
        print("✗ System Status: INCOMPLETE")
        print("  Some required files are missing or invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
