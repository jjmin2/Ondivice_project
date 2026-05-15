"""
Audio Transcription Module
비디오에서 오디오 추출 및 Whisper 기반 음성 전사
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import subprocess
import sys

try:
    import whisper
except ImportError:
    whisper = None


# ==============================
# 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent

VIDEOS_DIR = BASE_DIR / "videos"
AUDIO_TEMP_DIR = BASE_DIR / "outputs" / ".audio_temp"
TRANSCRIPTION_OUTPUT = BASE_DIR / "outputs" / "transcription_segments.json"

# Whisper 모델 (Jetson Nano 환경 고려)
WHISPER_MODEL_NAME = "base"  # tiny, base, small, medium 중 선택


# ==============================
# 데이터 구조
# ==============================
@dataclass
class SpeechSegment:
    """음성 세그먼트 (타임스탬프 포함)"""
    start: float
    end: float
    text: str
    language: str = "ko"


# ==============================
# 유틸 함수
# ==============================
def extract_audio_from_video(video_file: Path, output_audio: Path) -> bool:
    """
    비디오에서 오디오 추출 (ffmpeg 사용)
    
    Args:
        video_file: 입력 비디오 파일
        output_audio: 출력 오디오 파일
    
    Returns:
        성공 여부
    """
    video_file = Path(video_file)
    output_audio = Path(output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    
    if not video_file.exists():
        print(f"✗ Video file not found: {video_file}")
        return False
    
    try:
        # FFmpeg 경로 지정 (설치된 경로)
        ffmpeg_path = r"C:\temp\ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
        
        # ffmpeg 커맨드: 비디오에서 오디오만 추출
        cmd = [
            ffmpeg_path,
            "-i", str(video_file),
            "-q:a", "9",  # 오디오 품질 (낮을수록 좋음)
            "-y",  # 기존 파일 덮어쓰기
            str(output_audio)
        ]
        
        print(f"[ffmpeg] Extracting audio from {video_file.name}...")
        
        # ffmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ ffmpeg error: {result.stderr}")
            return False
        
        if output_audio.exists():
            size_mb = output_audio.stat().st_size / (1024 * 1024)
            print(f"✓ Audio extracted: {output_audio.name} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"✗ Audio file not created")
            return False
            
    except FileNotFoundError:
        print("✗ FFmpeg not found. Please install FFmpeg:")
        print("   1. Download from: https://ffmpeg.org/download.html")
        print("   2. Or run: pip install ffmpeg-python")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def transcribe_audio(
    audio_file: Path,
    model_name: str = "base",
    language: str = "ko"
) -> List[SpeechSegment]:
    """
    Whisper로 오디오 파일 전사
    
    Args:
        audio_file: 오디오 파일 경로
        model_name: Whisper 모델 (tiny/base/small/medium)
        language: 언어 코드 (ko=Korean, en=English, etc.)
    
    Returns:
        음성 세그먼트 리스트
    """
    audio_file = Path(audio_file)
    
    if whisper is None:
        print("✗ Whisper not installed")
        print("   Install with: pip install openai-whisper")
        return []
    
    if not audio_file.exists():
        print(f"✗ Audio file not found: {audio_file}")
        return []
    
    try:
        # Whisper 모델 로드
        print(f"[Whisper] Loading model: {model_name}")
        model = whisper.load_model(model_name)
        
        # 음성 전사
        print(f"[Whisper] Transcribing {audio_file.name}...")
        result = model.transcribe(
            str(audio_file),
            language=language,
            verbose=False
        )
        
        # 세그먼트 파싱
        segments = []
        for seg in result["segments"]:
            segments.append(
                SpeechSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"].strip(),
                    language=result.get("language", language)
                )
            )
        
        print(f"✓ Transcribed {len(segments)} segments")
        print(f"  Detected language: {result.get('language', 'unknown')}")
        
        return segments
        
    except Exception as e:
        print(f"✗ Transcription error: {e}")
        return []


def save_transcription(path: Path, segments: List[SpeechSegment]) -> bool:
    """음성 세그먼트를 JSON으로 저장"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data = [asdict(s) for s in segments]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved to: {path}")
        return True
    except Exception as e:
        print(f"✗ Error saving transcription: {e}")
        return False


def find_video_file(directory: Path) -> Optional[Path]:
    """디렉토리에서 첫 번째 비디오 파일 찾기"""
    directory = Path(directory)
    
    if not directory.exists():
        return None
    
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    
    for ext in video_extensions:
        video_files = list(directory.glob(f"*{ext}"))
        if video_files:
            return video_files[0]
    
    return None


def find_audio_file(directory: Path) -> Optional[Path]:
    """디렉토리에서 첫 번째 오디오 파일 찾기"""
    directory = Path(directory)
    
    if not directory.exists():
        return None
    
    audio_extensions = [".wav", ".mp3", ".m4a", ".flac", ".aac"]
    
    for ext in audio_extensions:
        audio_files = list(directory.glob(f"*{ext}"))
        if audio_files:
            return audio_files[0]
    
    return None


# ==============================
# 메인 함수
# ==============================
def main(
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    model_name: str = "base",
    language: str = "ko",
    skip_extraction: bool = False
) -> List[SpeechSegment]:
    """
    End-to-end 음성 전사 파이프라인
    
    Args:
        input_file: 비디오 또는 오디오 파일 (미지정시 자동 탐색)
        output_file: 출력 JSON 경로 (기본값: TRANSCRIPTION_OUTPUT)
        model_name: Whisper 모델
        language: 음성 언어 (ko/en/etc.)
        skip_extraction: True이면 오디오 추출 스킵
    
    Returns:
        음성 세그먼트 리스트
    """
    output_file = Path(output_file or TRANSCRIPTION_OUTPUT)
    
    # 입력 파일 결정
    if input_file is None:
        print("[Auto-detect] Searching for media files...")
        
        # 오디오 파일 먼저 찾기
        audio_file = find_audio_file(VIDEOS_DIR)
        if audio_file:
            print(f"✓ Found audio: {audio_file.name}")
            input_file = audio_file
            skip_extraction = True
        else:
            # 비디오 파일 찾기
            video_file = find_video_file(VIDEOS_DIR)
            if video_file:
                print(f"✓ Found video: {video_file.name}")
                input_file = video_file
            else:
                print("✗ No video or audio files found in:", VIDEOS_DIR)
                return []
    
    input_file = Path(input_file)
    
    # 오디오 파일 준비
    if skip_extraction or input_file.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac"]:
        audio_file = input_file
        print(f"Using audio file: {audio_file.name}")
    else:
        # 비디오에서 오디오 추출
        audio_file = AUDIO_TEMP_DIR / f"{input_file.stem}_audio.wav"
        success = extract_audio_from_video(input_file, audio_file)
        if not success:
            return []
    
    # Whisper 음성 전사
    print(f"\n[Whisper] Starting transcription...")
    segments = transcribe_audio(audio_file, model_name, language)
    
    if not segments:
        print("✗ No segments extracted")
        return []
    
    # 결과 저장
    print(f"\n[Save] Writing output...")
    save_transcription(output_file, segments)
    
    return segments


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Transcription with Whisper")
    parser.add_argument("--input", "-i", type=str, help="Input video or audio file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--model", "-m", type=str, default="base", 
                       help="Whisper model (tiny/base/small/medium)")
    parser.add_argument("--language", "-l", type=str, default="ko",
                       help="Language code (ko/en/etc.)")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip audio extraction (input is audio)")
    
    args = parser.parse_args()
    
    try:
        segments = main(
            input_file=args.input,
            output_file=args.output,
            model_name=args.model,
            language=args.language,
            skip_extraction=args.skip_extraction
        )
        print(f"\n✓ Transcription complete: {len(segments)} segments")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
