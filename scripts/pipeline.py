"""
End-to-End Multimodal Video Edit Point Extraction Pipeline
온디바이스 멀티모달 영상 편집포인트 추출 통합 파이프라인

Pipeline Flow:
1. Pose 추출 (pose_analyzer.py)
2. Motion event 추출 (extract_motion_events.py)
3. 음성 전사 (transcribe_audio.py)
4. 멀티모달 병합 및 highlight 추출 (merge_whisper_pose.py)
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 모듈 임포트
try:
    from extract_motion_events import extract_motion_events
    from transcribe_audio import main as transcribe_main
    from merge_whisper_pose import main as merge_main
except ImportError as e:
    print(f"Warning: Some modules not imported: {e}")


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
HIGHLIGHT_REEL_FILE = OUTPUTS_DIR / "highlight_reel.mp4"


# ==============================
# 유틸 함수
# ==============================
def print_section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def print_success(msg: str):
    print(f"✓ {msg}")

def print_error(msg: str):
    print(f"✗ {msg}")


def format_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def highlight_labels(event_type: str) -> Dict[str, str]:
    motion_map = {
        "emphatic_gesture": "Strong Gesture",
        "emphasized_speech": "Normal Gesture",
        "emphasized_speech_with_gesture": "Strong Gesture",
    }
    speech_map = {
        "emphatic_gesture": "Normal",
        "emphasized_speech": "Emphasized",
        "emphasized_speech_with_gesture": "Emphasized",
    }
    return {
        "motion": motion_map.get(event_type, event_type),
        "speech": speech_map.get(event_type, "Normal")
    }


def check_dependencies():
    """필요한 라이브러리 및 도구 확인"""
    print("Checking dependencies...")
    
    dependencies = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "whisper": "openai-whisper",
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (install: pip install {package})")
            missing.append(package)

    # FFmpeg 확인 (비디오 하이라이트 생성에 필요)
    try:
        import shutil
        if shutil.which("ffmpeg"):
            print("  ✓ ffmpeg")
        else:
            print("  ✗ ffmpeg (install FFmpeg and ensure it is on PATH)")
            missing.append("ffmpeg")
    except Exception:
        print("  ✗ ffmpeg (install FFmpeg and ensure it is on PATH)")
        missing.append("ffmpeg")
    
    if missing:
        print(f"\nMissing packages/tools: {', '.join(sorted(set(missing)))}")
        return False
    
    return True


def get_latest_pose_data(directory: Path) -> Optional[Path]:
    """최신 pose_data JSON 파일 찾기"""
    directory = Path(directory)
    pose_files = list(directory.glob("pose_data*.json"))
    
    if not pose_files:
        return None
    
    # 타임스탐프 기준으로 정렬하여 최신 파일 반환
    return sorted(pose_files)[-1]


def save_pipeline_log(log_data: Dict[str, Any], output_file: Path):
    """파이프라인 실행 로그 저장"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


# ==============================
# 파이프라인 단계
# ==============================
def step0_extract_pose(video_file: Path, log: Dict[str, Any] = None) -> bool:
    """
    Step 0: Pose 데이터 추출 (비디오에서)
    
    Requires:
    - 비디오 파일
    
    Outputs:
    - pose_data.json
    """
    print_section("Step 0: Extract Pose Data from Video")
    
    try:
        video_file = Path(video_file)
        if not video_file.exists():
            print_error(f"Video file not found: {video_file}")
            return False
        
        print(f"Input video: {video_file.name}")
        print("This will open a webcam window for pose detection...")
        print("Press 'q' to stop and save pose data")
        
        # pose_analyzer.py 실행 (별도 프로세스로)
        import subprocess
        result = subprocess.run([
            "python3", "scripts/pose_analyzer.py", str(video_file)
        ], capture_output=True, text=True, cwd=BASE_DIR)
        
        if result.returncode != 0:
            print_error(f"Pose extraction failed: {result.stderr}")
            return False
        
        # 생성된 pose_data 파일 확인
        pose_files = list(OUTPUTS_DIR.glob("pose_data*.json"))
        if not pose_files:
            print_error("No pose_data.json file was created")
            return False
        
        latest_pose_file = sorted(pose_files)[-1]
        print_success(f"Pose data saved: {latest_pose_file.name}")
        
        if log is not None:
            log["step0_pose_extraction"] = {
                "status": "success",
                "input_video": str(video_file),
                "output_file": str(latest_pose_file)
            }
        
        return True
        
    except Exception as e:
        print_error(f"Pose extraction failed: {e}")
        if log is not None:
            log["step0_pose_extraction"]["status"] = "failed"
            log["step0_pose_extraction"]["error"] = str(e)
        return False


def step1_extract_motion_events(log: Dict[str, Any] = None) -> bool:
    """
    Step 1: Motion event 추출
    
    Requires:
    - pose_data JSON 파일 (step0에서 생성됨)
    
    Outputs:
    - motion_events.json
    """
    print_section("Step 1: Extract Motion Events")
    
    try:
        # 최신 pose_data 파일 찾기
        pose_data_file = get_latest_pose_data(OUTPUTS_DIR)
        if not pose_data_file:
            print_error("No pose_data.json found")
            print("Please run Step 0 first to generate pose data")
            return False
        
        motion_events_file = OUTPUTS_DIR / "motion_events.json"
        
        # Motion event 추출
        print(f"Input: {pose_data_file.name}")
        print(f"Output: {motion_events_file.name}")
        
        count = extract_motion_events(pose_data_file, motion_events_file)
        
        print_success(f"Extracted {count} motion events")
        
        if log is not None:
            log["step1_motion_events"] = {
                "status": "success",
                "events_count": count,
                "input_file": str(pose_data_file),
                "output_file": str(motion_events_file)
            }
        
        return True
        
    except Exception as e:
        print_error(f"Motion event extraction failed: {e}")
        if log is not None:
            log["step1_motion_events"]["status"] = "failed"
            log["step1_motion_events"]["error"] = str(e)
        return False


def step2_transcribe_audio(
    video_file: Optional[Path] = None,
    audio_file: Optional[Path] = None,
    model_name: str = "base",
    language: str = "ko",
    log: Dict[str, Any] = None
) -> bool:
    """
    Step 2: 음성 전사 (Whisper)
    
    Requires:
    - 오디오 또는 비디오 파일
    - FFmpeg (비디오에서 오디오 추출 시)
    
    Outputs:
    - transcription_segments.json
    """
    print_section("Step 2: Transcribe Audio with Whisper")
    
    try:
        print(f"Model: {model_name}")
        print(f"Language: {language}")
        
        segments = transcribe_main(
            input_file=audio_file or video_file,
            model_name=model_name,
            language=language,
            skip_extraction=False
        )
        
        if not segments:
            print_error("No speech segments extracted")
            return False
        
        print_success(f"Transcribed {len(segments)} speech segments")
        
        if log is not None:
            log["step2_transcription"] = {
                "status": "success",
                "segments_count": len(segments),
                "output_file": str(OUTPUTS_DIR / "transcription_segments.json")
            }
        
        return True
        
    except Exception as e:
        print_error(f"Transcription failed: {e}")
        if log is not None:
            log["step2_transcription"]["status"] = "failed"
            log["step2_transcription"]["error"] = str(e)
        return False


def step3_merge_multimodal(
    model_name: str = "base",
    log: Dict[str, Any] = None
) -> list:
    """
    Step 3: 멀티모달 병합 및 Highlight 추출
    
    Requires:
    - motion_events.json (Step 1)
    - transcription_segments.json (Step 2)
    
    Outputs:
    - highlight_segments.json (최종 편집 포인트)
    """
    print_section("Step 3: Merge Multimodal & Extract Highlights")
    
    try:
        # 필요한 파일 확인
        motion_events_file = OUTPUTS_DIR / "motion_events.json"
        
        if not motion_events_file.exists():
            print_error(f"Motion events file not found: {motion_events_file}")
            print("Please run Step 1 first")
            return []
        
        # Merge 실행
        highlights = merge_main(model_name=model_name)
        
        if not highlights:
            print_error("No highlights generated")
            return []
        
        print_success(f"Generated {len(highlights)} highlight segments")
        
        highlights.sort(key=lambda h: h.start)
        
        # 사용자 친화적 요약 출력
        print("\n🔥 Highlight Summary\n")
        for i, h in enumerate(highlights, 1):
            labels = highlight_labels(h.event_type)
            
            # gesture tag 문자열 생성
            gesture_text = ""
            if h.gesture_tags:
                gesture_text = f" ({', '.join(h.gesture_tags)})"
            
            print(f"\n🔥 Highlight #{i}")
            print(f"⏱ {format_time(h.start)} ~ {format_time(h.end)}")
            print(f"🗣 \"{h.text.strip()}\"")
            print(f"\n📈 Highlight Score: {h.highlight_score:.3f}")
            print(f"🤲 Motion: {labels['motion']}{gesture_text}")
            print(f"🎤 Speech: {labels['speech']}")
        
        if log is not None:
            log["step3_highlights"] = {
                "status": "success",
                "highlights_count": len(highlights),
                "output_file": str(OUTPUTS_DIR / "highlight_segments.json"),
                "top_highlights": [
                    {
                        "start": h.start,
                        "end": h.end,
                        "highlight_score": h.highlight_score,
                        "event_type": h.event_type
                    }
                    for h in highlights[:5]
                ]
            }
        
        return highlights
        
    except Exception as e:
        print_error(f"Highlight extraction failed: {e}")
        if log is not None:
            log["step3_highlights"]["status"] = "failed"
            log["step3_highlights"]["error"] = str(e)
        return []


# ==============================
# 메인 파이프라인
# ==============================
def run_full_pipeline(
    video_file: Optional[Path] = None,
    audio_file: Optional[Path] = None,
    whisper_model: str = "base",
    language: str = "ko",
    skip_step1: bool = False,
) -> bool:
    """
    전체 파이프라인 실행
    
    Args:
        video_file: 비디오 파일 경로 (pose data 생성용)
        audio_file: 오디오/비디오 파일 경로 (음성 전사용)
        whisper_model: Whisper 모델 (tiny/base/small/medium)
        language: 음성 언어 (ko/en/etc.)
        skip_step1: Step 1 (motion event) 스킵
    
    Returns:
        성공 여부
    """
    print("\n" + "🎬" * 30)
    print("  OnDevice Multimodal Video Edit Point Extraction Pipeline")
    print("🎬" * 30)
    
    # 입력 경로 정규화
    if video_file:
        video_file = Path(video_file)
    if audio_file:
        audio_file = Path(audio_file)

    # 로그 초기화
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "whisper_model": whisper_model,
            "language": language,
            "skip_step1": skip_step1
        },
        "steps": {}
    }
    
    # 의존성 확인
    print("\n[Dependency Check]")
    if not check_dependencies():
        print_error("Missing required packages")
        return False
    
    # Step 0: Pose 데이터 생성 (비디오에서)
    if video_file and not skip_step1:
        log_data["steps"]["step0_pose_extraction"] = {"status": "pending"}
        if not step0_extract_pose(video_file, log_data["steps"]):
            print_error("Pipeline failed at Step 0")
            return False
    
    # Step 1: Motion event 추출
    if not skip_step1:
        log_data["steps"]["step1_motion_events"] = {"status": "pending"}
        if not step1_extract_motion_events(log_data["steps"]):
            print_error("Pipeline failed at Step 1")
            return False
    else:
        print_section("Step 1: Skipped (using existing motion_events.json)")
    
    # Step 2: Whisper 음성 전사
    log_data["steps"]["step2_transcription"] = {"status": "pending"}
    if not step2_transcribe_audio(video_file, audio_file, whisper_model, language, log_data["steps"]):
        print_error("Pipeline failed at Step 2")
        return False
    
    # Step 3: Multimodal 병합 및 highlight 추출
    log_data["steps"]["step3_highlights"] = {"status": "pending"}
    highlights = step3_merge_multimodal(whisper_model, log_data["steps"])
    if not highlights:
        print_error("Pipeline failed at Step 3")
        return False

    # Step 4: Highlight video 추출
    if video_file:
        log_data["steps"]["step4_export_video"] = {"status": "pending"}
        highlight_video_file = OUTPUTS_DIR / f"{video_file.stem}_highlight_reel.mp4"
        if not step4_export_highlight_video(video_file, highlights, highlight_video_file, log_data["steps"]):
            print_error("Pipeline failed at Step 4")
            return False
    
    # 최종 결과
    print_section("Pipeline Complete ✓")
    
    output_file = OUTPUTS_DIR / "highlight_segments.json"
    if output_file.exists():
        print(f"\nFinal output: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            highlights = json.load(f)
        print(f"Total highlights: {len(highlights)}")
        print(f"\nUsage:")
        print(f"  - Import highlights into your video editor")
        print(f"  - Use for automatic chapter generation")
        print(f"  - Fast-track editing workflow")
        if video_file:
            highlight_video_file = OUTPUTS_DIR / f"{video_file.stem}_highlight_reel.mp4"
            if highlight_video_file.exists():
                print(f"  - Highlight reel: {highlight_video_file}")
    
    # 파이프라인 로그 저장
    log_file = OUTPUTS_DIR / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_pipeline_log(log_data, log_file)
    print(f"\nPipeline log saved: {log_file.name}")
    
    return True


# ==============================
# CLI
# ==============================
def step4_export_highlight_video(
    video_file: Path,
    highlights: list,
    output_file: Path,
    log: Dict[str, Any] = None
) -> bool:
    """
    Step 4: Highlight video 파일 생성
    """
    print_section("Step 4: Export Highlight Video")

    try:
        video_file = Path(video_file)
        if not video_file.exists():
            print_error(f"Video file not found: {video_file}")
            return False
        if not highlights:
            print_error("No highlight segments to export")
            return False

        tmp_dir = OUTPUTS_DIR / ".highlight_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        list_file = tmp_dir / "highlight_concat.txt"
        clip_paths = []

        highlights = sorted(
            highlights,
            key=lambda h: h.start if hasattr(h, "start") else h["start"]
        )

        for idx, seg in enumerate(highlights, start=1):
            start = float(seg.start) if hasattr(seg, "start") else float(seg["start"])
            end = float(seg.end) if hasattr(seg, "end") else float(seg["end"])
            duration = max(end - start, 0.1)
            clip_path = tmp_dir / f"highlight_part_{idx:02d}.mp4"

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start:.3f}",
                "-i", str(video_file),
                "-t", f"{duration:.3f}",
                "-c", "copy",
                str(clip_path)
            ]

            print(f"Extracting clip {idx}: {start:.2f}s -> {end:.2f}s")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print_error(f"FFmpeg clip extraction failed: {result.stderr.strip()}")
                return False

            clip_paths.append(clip_path)

        with open(list_file, "w", encoding="utf-8") as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path.as_posix()}'\n")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_file)
        ]

        print(f"Building highlight reel: {output_file.name}")
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print_error(f"FFmpeg concat failed: {result.stderr.strip()}")
            return False

        if log is not None:
            log["step4_export_video"] = {
                "status": "success",
                "output_file": str(output_file)
            }

        print_success(f"Highlight reel saved: {output_file.name}")
        return True

    except FileNotFoundError:
        print_error("FFmpeg not found. Install FFmpeg and ensure it is on PATH.")
        if log is not None:
            log["step4_export_video"] = {
                "status": "failed",
                "error": "ffmpeg not found"
            }
        return False
    except Exception as e:
        print_error(f"Highlight video export failed: {e}")
        if log is not None:
            log["step4_export_video"] = {
                "status": "failed",
                "error": str(e)
            }
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OnDevice Multimodal Video Edit Point Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python pipeline.py --video video.mp4
  
  # With specific audio file
  python pipeline.py --video video.mp4 --audio audio.wav
  
  # Skip motion extraction (use existing motion_events.json)
  python pipeline.py --audio audio.wav --skip-step1
  
  # With different Whisper model
  python pipeline.py --video video.mp4 --model tiny
        """
    )
    
    parser.add_argument("--video", "-v", type=str,
                       help="Video file path")
    parser.add_argument("--audio", "-a", type=str,
                       help="Audio file path (overrides video)")
    parser.add_argument("--model", "-m", type=str, default="base",
                       choices=["tiny", "base", "small", "medium"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--language", "-l", type=str, default="ko",
                       help="Speech language code (default: ko)")
    parser.add_argument("--skip-step1", action="store_true",
                       help="Skip motion event extraction")
    
    args = parser.parse_args()
    
    try:
        success = run_full_pipeline(
            video_file=args.video,
            audio_file=args.audio,
            whisper_model=args.model,
            language=args.language,
            skip_step1=args.skip_step1
        )
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print_error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
