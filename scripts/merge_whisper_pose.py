import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from dataclasses import field

try:
    import whisper
except ImportError:
    print("Warning: whisper not installed. Install with: pip install openai-whisper")
    whisper = None


# ==============================
# 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent

# Default paths (can be overridden)
MOTION_EVENTS_FILE = BASE_DIR / "outputs" / "motion_events.json"
HIGHLIGHT_OUTPUT_FILE = BASE_DIR / "outputs" / "highlight_segments.json"

# Whisper 모델: Jetson 환경이면 tiny 또는 base 권장
WHISPER_MODEL_NAME = "base"

# 하이라이트 판단 가중치
SPEECH_WEIGHT = 0.6
MOTION_WEIGHT = 0.4

# 키워드 (한글 + 영문)
KEYWORDS = [
    "중요", "핵심", "정리", "포인트", "주의", "기억", "결론", "요약",
    "important", "key", "summary", "point", "note", "remember", "conclusion"
]

# 세그먼트 병합 허용 간격(초)
MERGE_GAP_SEC = 0.5


# ==============================
# 데이터 구조
# ==============================
@dataclass
class WhisperSegment:
    """Whisper 오디오 전사 세그먼트"""
    start: float
    end: float
    text: str
    speech_score: float = 0.0


@dataclass
class MotionEvent:
    """Extract된 motion event"""
    start: float
    end: float
    duration: float
    action: str = "high_motion"
    average_score: float = 0.0
    peak_score: float = 0.0
    gesture_tags: List[str] = field(default_factory=list)


@dataclass
class HighlightSegment:
    """최종 highlight segment (편집 포인트)"""
    start: float
    end: float
    duration: float
    text: str
    speech_score: float
    motion_score: float
    highlight_score: float
    event_type: str
    gesture_tags: List[str] = field(default_factory=list)
    note: str =""



# ==============================
# 유틸 함수
# ==============================
def load_motion_events(path: Path) -> List[MotionEvent]:
    """motion_events.json 로드 및 파싱"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motion events file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []
    for item in data:
        events.append(
            MotionEvent(
                start=float(item["start"]),
                end=float(item["end"]),
                duration=float(item.get("duration", 0.0)),
                action=item.get("action", "high_motion"),
                average_score=float(item.get("average_score", 0.0)),
                peak_score=float(item.get("peak_score", 0.0)),
                gesture_tags=item.get("gesture_tags", [])
            )
        )
    return events


def transcribe_audio(audio_file: Path, model_name: str = "base") -> List[WhisperSegment]:
    """Whisper로 오디오 파일 전사 및 타임스탐프 생성"""
    audio_file = Path(audio_file)
    
    if whisper is None:
        raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
    
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    print(f"[Whisper] Loading model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"[Whisper] Transcribing audio: {audio_file.name}")
    result = model.transcribe(str(audio_file))

    segments = []
    for seg in result["segments"]:
        segments.append(
            WhisperSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
        )
    return segments



def compute_speech_score(text: str) -> float:
    """
    텍스트 기반 음성 강조도 점수 계산
    - 키워드 포함 여부: +0.6
    - 적당한 길이: +0.2~0.4
    """
    score = 0.0

    # 키워드 포함 가점
    for kw in KEYWORDS:
        if kw.lower() in text.lower():
            score += 0.6
            break  # 한 번만 가산

    # 텍스트 길이 기반 점수
    length = len(text.strip())
    if length >= 50:
        score += 0.4
    elif length >= 20:
        score += 0.2

    return min(score, 1.0)


def compute_motion_score(event: MotionEvent) -> float:
    """
    Motion event 기반 모션 강조도 점수 계산
    - peak_score 직접 사용 (있으면)
    - 없으면 duration 기반 fallback
    """
    # peak_score가 있으면 우선 사용
    if event.peak_score > 0:
        return min(event.peak_score, 1.0)
    
    # average_score도 고려
    if event.average_score > 0:
        return min(event.average_score * 1.2, 1.0)  # 약간의 부스트
    
    # 모두 없으면 duration 기반
    duration = event.end - event.start
    if duration >= 3.0:
        return 0.8
    elif duration >= 2.0:
        return 0.6
    elif duration >= 1.0:
        return 0.4
    else:
        return 0.2


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """두 구간의 겹치는 시간 계산 (초)"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_segments(segments: List[HighlightSegment], gap_sec: float = 0.5) -> List[HighlightSegment]:
    """가까운 highlight segment 병합"""
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x.start)
    merged = [segments[0]]

    for seg in segments[1:]:
        last = merged[-1]
        
        # gap_sec 이내이면 병합
        if seg.start <= last.end + gap_sec:
            last.end = max(last.end, seg.end)
            last.duration = last.end - last.start
            
            # 텍스트 병합
            if seg.text and last.text:
                last.text += " " + seg.text
            elif seg.text:
                last.text = seg.text
            
            # 점수는 최댓값 사용
            last.speech_score = max(last.speech_score, seg.speech_score)
            last.motion_score = max(last.motion_score, seg.motion_score)
            last.highlight_score = max(last.highlight_score, seg.highlight_score)
            last.note = last.note + " | " + seg.note
        else:
            merged.append(seg)

    return merged



# ==============================
# 하이라이트 생성 로직
# ==============================
def build_highlights(
    whisper_segments: List[WhisperSegment],
    motion_events: List[MotionEvent],
) -> List[HighlightSegment]:
    """
    Whisper 세그먼트와 Motion event를 결합하여 highlight 생성
    
    Highlight score = speech_score * SPEECH_WEIGHT + motion_score * MOTION_WEIGHT + overlap_bonus
    """
    candidates: List[HighlightSegment] = []

    for s in whisper_segments:
        s.speech_score = compute_speech_score(s.text)

        for m in motion_events:
            # 시간 겹침 확인
            ov = overlap_duration(s.start, s.end, m.start, m.end)
            if ov <= 0:
                continue

            # 모션 점수 계산
            m_score = compute_motion_score(m)

            # Multimodal score 계산
            speech_component = SPEECH_WEIGHT * s.speech_score
            motion_component = MOTION_WEIGHT * m_score
            
            # 겹침 비율에 따른 보너스 (최대 0.15)
            overlap_ratio = min(ov / max(s.end - s.start, 0.001), 1.0)
            overlap_bonus = overlap_ratio * 0.15

            highlight_score = min(speech_component + motion_component + overlap_bonus, 1.0)

            # 점수 임계값: 0.4 이상만 후보
            if highlight_score >= 0.4:
                # 이벤트 타입 결정
                event_type = "emphasized_speech"
                if s.speech_score >= 0.6 and m_score >= 0.5:
                    event_type = "emphasized_speech_with_gesture"
                elif m_score >= 0.6:
                    event_type = "emphatic_gesture"

                # 주석 생성
                note_parts = []
                if s.speech_score >= 0.5:
                    note_parts.append(f"speech({s.speech_score:.2f})")
                if m_score >= 0.5:
                    note_parts.append(f"motion({m_score:.2f})")
                if ov > 0.5:
                    note_parts.append(f"overlap({ov:.2f}s)")

                candidates.append(
                    HighlightSegment(
                        start=min(s.start, m.start),
                        end=max(s.end, m.end),
                        duration=max(s.end, m.end) - min(s.start, m.start),
                        text=s.text,
                        speech_score=round(s.speech_score, 3),
                        motion_score=round(m_score, 3),
                        highlight_score=round(highlight_score, 3),
                        event_type=event_type,
                        gesture_tags=m.gesture_tags if hasattr(m, "gesture_tags") else [],
                        note=" | ".join(note_parts) if note_parts else "highlight"
                    )
                )

    # 세그먼트 병합
    merged = merge_segments(candidates, gap_sec=MERGE_GAP_SEC)

    # Highlight score 기준 정렬 (내림차순)
    merged = sorted(merged, key=lambda x: (-x.highlight_score, x.start))
    
    return merged


def save_highlights(path: Path, highlights: List[HighlightSegment]) -> None:
    """Highlight segments를 JSON 파일로 저장"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [asdict(h) for h in highlights]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==============================
# 메인 함수
# ==============================
def main(
    audio_file: Optional[Path] = None,
    motion_events_file: Optional[Path] = None,
    transcription_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    model_name: str = "base",
) -> List[HighlightSegment]:
    """
    End-to-end multimodal highlight extraction
    
    Args:
        audio_file: 오디오 파일 경로
        motion_events_file: motion events JSON 경로 (기본값: MOTION_EVENTS_FILE)
        transcription_file: 기존 transcription segments JSON 경로 (있으면 사용)
        output_file: 출력 JSON 경로 (기본값: HIGHLIGHT_OUTPUT_FILE)
        model_name: Whisper 모델 이름 (tiny/base/small/medium)
    
    Returns:
        List of HighlightSegment
    """
    # 경로 설정
    motion_events_file = Path(motion_events_file or MOTION_EVENTS_FILE)
    output_file = Path(output_file or HIGHLIGHT_OUTPUT_FILE)

    # 1. Motion events 로드
    print(f"\n[Step 1/3] Loading motion events...")
    try:
        motion_events = load_motion_events(motion_events_file)
        print(f"  ✓ Loaded {len(motion_events)} motion events")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return []

    # 2. Whisper 음성 전사 또는 기존 transcription 로드
    print(f"\n[Step 2/3] Loading speech segments...")
    
    # 기존 transcription 파일 사용 시도
    trans_file = Path(transcription_file or BASE_DIR / "outputs" / "transcription_segments.json")
    if trans_file.exists():
        print(f"  Using existing transcription: {trans_file.name}")
        try:
            with open(trans_file, "r", encoding="utf-8") as f:
                trans_data = json.load(f)
            whisper_segments = []
            for item in trans_data:
                whisper_segments.append(
                    WhisperSegment(
                        start=float(item["start"]),
                        end=float(item["end"]),
                        text=item["text"],
                        speech_score=float(item.get("speech_score", 0.0))
                    )
                )
            print(f"  ✓ Loaded {len(whisper_segments)} speech segments")
        except Exception as e:
            print(f"  ✗ Error loading transcription: {e}")
            # 새로운 오디오 파일에서 transcription 생성 시도
            if audio_file is None:
                return []
    else:
        # 새로운 오디오 파일에서 transcription 생성
        if audio_file is None:
            audio_dir = BASE_DIR / "videos"
            audio_extensions = [".wav", ".mp3", ".m4a", ".flac"]
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(audio_dir.glob(f"*{ext}"))
            if not audio_files:
                print(f"  ✗ No audio files found in {audio_dir}")
                return []
            audio_file = audio_files[0]
            print(f"  Found audio: {audio_file.name}")
        
        try:
            whisper_segments = transcribe_audio(audio_file, model_name)
            print(f"  ✓ Transcribed {len(whisper_segments)} speech segments")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []

    # 3. Highlight 생성
    print(f"\n[Step 3/3] Building multimodal highlights...")
    highlights = build_highlights(whisper_segments, motion_events)
    print(f"  ✓ Generated {len(highlights)} highlight segments")

    # 저장
    save_highlights(output_file, highlights)
    print(f"\n✓ Highlights saved to: {output_file}")
    
    return highlights


if __name__ == "__main__":
    import sys
    
    # 커맨드라인 인자 처리
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    try:
        highlights = main(audio_file=audio_file, model_name=model_name)
        print(f"\n총 {len(highlights)}개의 편집 포인트 추출 완료")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
