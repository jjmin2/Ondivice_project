# OnDevice Multimodal Video Edit Point Extraction System
## 온디바이스 멀티모달 영상 편집포인트 추출 시스템

### 📋 프로젝트 개요

이 프로젝트는 **NVIDIA Jetson Orin Nano** 기반의 경량 온디바이스 시스템으로,  
영상 속 사람의 **행동(모션)과 음성 정보**를 동시에 분석하여  
자동으로 **의미 있는 편집 포인트(highlight)**를 추출합니다.

**목표**: 긴 영상을 자동으로 분석하고 강조 구간을 추출하여  
사용자가 효율적으로 편집할 수 있도록 돕는 시스템

---

## 🎯 핵심 기능

### 1. **모션 분석 (Motion Analysis)**
- MediaPipe Pose를 이용한 전신 landmark 추출
- 프레임 간 움직임 변화량(movement_score) 계산
- 이벤트 세그멘테이션 (threshold, grace period, min_duration)

### 2. **음성 분석 (Speech Analysis)**
- OpenAI Whisper를 이용한 음성 전사
- 타임스탬프 포함 세그먼트 자동 생성
- 한국어, 영어 등 다국어 지원

### 3. **멀티모달 정렬 (Multimodal Alignment)**
- Motion events와 Speech segments의 시간축 정렬
- 시간 겹침(overlap) 기반 연관성 계산

### 4. **Highlight 추출 (Highlight Extraction)**
```
Highlight Score =
    Speech Weight(0.6) × Speech Score
    + Motion Weight(0.4) × Motion Score
    + Overlap Bonus (최대 0.15)

※ 최종 점수 0.4 이상인 구간만 편집 포인트로 선정
```

### 5. **편집포인트 출력 (Edit Point Output)**
```json
{
    "start": 12.3,
    "end": 15.2,
    "duration": 2.9,
    "text": "이 부분이 매우 중요합니다.",
    "speech_score": 0.750,
    "motion_score": 0.420,
    "highlight_score": 0.768,
    "event_type": "emphasized_speech_with_gesture",
    "note": "speech(0.75) | motion(0.42) | overlap(1.50s)"
}
```

### 6. **Highlight Reel 생성 (Video Export)**
- FFmpeg를 이용해 편집 포인트 클립을 자동 추출·연결
- `{영상명}_highlight_reel.mp4` 파일 자동 생성

---

## 🏗️ 시스템 아키텍처

### 모듈 구조

```
scripts/
├── pose_analyzer.py          # ① 웹캠/비디오에서 pose 데이터 추출
├── extract_motion_events.py  # ② Pose 데이터에서 motion event 추출
├── transcribe_audio.py       # ③ 오디오 파일을 Whisper로 전사
├── merge_whisper_pose.py     # ④ Motion + Speech를 결합하여 highlight 추출
├── visualize_pose.py         # 시각화 도구
├── pipeline.py               # ⑤ End-to-end 통합 파이프라인
└── validate.py               # 시스템 검증 및 테스트
```

### 데이터 흐름

```
Video/Webcam
    ↓
pose_analyzer.py → pose_data_{timestamp}.json
    ↓
extract_motion_events.py → motion_events.json
    ↓ (병렬)              ↓
    ↓                transcribe_audio.py → transcription_segments.json
    ↓                     ↓
    └─────────────────────┘
            ↓
    merge_whisper_pose.py
            ↓
    highlight_segments.json (최종 편집포인트)
            ↓
    FFmpeg (Step 4)
            ↓
    {영상명}_highlight_reel.mp4 + pipeline_log_{timestamp}.json
```

---

## 🔍 편집 포인트 선정 기준

### 음성 점수 (Speech Score)
| 조건 | 점수 |
|------|------|
| 강조 키워드 포함 | +0.6 |
| 텍스트 50자 이상 | +0.4 |
| 텍스트 20자 이상 | +0.2 |

**강조 키워드**: 중요, 핵심, 정리, 포인트, 주의, 기억, 결론, 요약, important, key, summary, point, note, remember, conclusion

### 모션 점수 (Motion Score)
| 조건 | 점수 |
|------|------|
| peak_score > 0 | peak_score 직접 사용 (최대 1.0) |
| average_score > 0 | average_score × 1.2 (부스트) |
| duration 3초 이상 | 0.8 |
| duration 2초 이상 | 0.6 |
| duration 1초 이상 | 0.4 |
| duration 1초 미만 | 0.2 |

### 이벤트 타입 분류
| 타입 | 조건 |
|------|------|
| `emphasized_speech_with_gesture` | speech_score ≥ 0.6 AND motion_score ≥ 0.5 |
| `emphatic_gesture` | motion_score ≥ 0.6 |
| `emphasized_speech` | 그 외 (기본) |

---

## 🚀 설치 및 환경 설정

### 요구사항
- **Python** 3.10+
- **NVIDIA Jetson Orin Nano** (또는 개발용 Linux/Windows/Mac)
- **GPU**: Jetson Orin Nano 내장 GPU 활용

### 의존성 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
matplotlib>=3.7.0
openai-whisper>=20231117
```

비디오 처리에 FFmpeg도 필요합니다:
```bash
# Windows
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
apt install ffmpeg
```

### 프로젝트 구조

```
ondivice/
├── scripts/
│   ├── pose_analyzer.py
│   ├── extract_motion_events.py
│   ├── transcribe_audio.py
│   ├── merge_whisper_pose.py
│   ├── visualize_pose.py
│   ├── pipeline.py
│   └── validate.py
├── videos/              # 입력 비디오 파일
├── outputs/            # 생성된 데이터
│   ├── pose_data_{timestamp}.json
│   ├── motion_events.json
│   ├── transcription_segments.json
│   ├── highlight_segments.json
│   ├── {영상명}_highlight_reel.mp4
│   └── pipeline_log_{timestamp}.json
└── venv/               # 가상 환경
```

---

## 📖 사용 방법

### 1️⃣ 전체 파이프라인 실행 (추천)

```bash
# 기본 사용 (모델: base, 언어: 한국어)
python scripts/pipeline.py --video videos/lecture.mp4

# 커스터마이징 옵션
python scripts/pipeline.py \
    --video videos/lecture.mp4 \
    --model tiny \
    --language ko \
    --skip-step1
```

**옵션 설명:**
| 옵션 | 단축키 | 설명 | 기본값 |
|------|--------|------|--------|
| `--video` | `-v` | 입력 비디오 파일 | - |
| `--audio` | `-a` | 오디오 파일 (선택사항, 비디오 우선) | - |
| `--model` | `-m` | Whisper 모델 크기 (tiny/base/small/medium) | base |
| `--language` | `-l` | 음성 언어 | ko |
| `--skip-step1` | - | Motion extraction 스킵 (기존 motion_events.json 사용) | false |

파이프라인 완료 후 `outputs/pipeline_log_{timestamp}.json`에 실행 로그가 저장됩니다.

### 2️⃣ 단계별 실행

#### Step 0: Pose 데이터 추출
```bash
python scripts/pose_analyzer.py
# 비디오/웹캠에서 실시간 pose 추출
# 'q' 키로 종료하면 pose_data_{timestamp}.json 생성
```

#### Step 1: Motion Event 추출
```bash
python scripts/extract_motion_events.py
# outputs/pose_data_*.json (최신 파일 자동 선택) → motion_events.json
```

#### Step 2: 음성 전사
```bash
python scripts/transcribe_audio.py \
    --input videos/lecture.mp4 \
    --model base \
    --language ko
# transcription_segments.json 생성
```

#### Step 3: Highlight 추출
```bash
python scripts/merge_whisper_pose.py
# motion_events.json + transcription_segments.json
# → highlight_segments.json (최종 편집포인트)
```

#### Step 4: Highlight Video 생성
`pipeline.py` 실행 시 자동으로 처리됩니다. 개별 실행은 `pipeline.py`를 통해서만 가능합니다.

### 3️⃣ 시스템 검증

```bash
# 시스템 상태 확인
python scripts/validate.py

# 샘플 데이터 생성 (테스트용)
python scripts/validate.py --create-sample
```

---

## ⚙️ 설정 가능한 파라미터

### extract_motion_events.py

```python
THRESHOLD = 0.05             # 움직임 감지 임계값 (낮을수록 민감)
FPS = 30                     # 프레임률
GRACE_FRAMES = int(0.7*FPS) # 관용 프레임 수 (짧은 정지 무시)
MIN_DURATION = 1.0           # 최소 이벤트 지속시간 (초)
MIN_PEAK_SCORE = 0.15        # 최소 피크 점수
```

### merge_whisper_pose.py

```python
SPEECH_WEIGHT = 0.6          # 음성 가중치
MOTION_WEIGHT = 0.4          # 모션 가중치
MERGE_GAP_SEC = 0.5          # 세그먼트 병합 허용 간격 (초)

KEYWORDS = [                 # 강조 감지 키워드
    "중요", "핵심", "정리", "포인트", "주의", "기억", "결론", "요약",
    "important", "key", "summary", "point", "note", "remember", "conclusion"
]
```

---

## 📊 출력 형식

### motion_events.json
```json
[
    {
        "start": 0.3,
        "end": 1.9,
        "duration": 1.6,
        "action": "high_motion",
        "average_score": 0.0554,
        "peak_score": 0.2213,
        "gesture_tags": [
            "hand_raise",
            "pointing"
        ]
    }
]
```

### transcription_segments.json
```json
[
    {
        "start": 2.94,
        "end": 6.2,
        "text": "변화하는 비율 변화율.",
        "language": "ko"
    }
]
```

### highlight_segments.json (최종)
```json
[
    {
        "start": 5.43,
        "end": 15.9,
        "duration": 10.47,
        "text": "이 변화율이라는 게 뭐라고 뭐냐면...",
        "speech_score": 0.4,
        "motion_score": 1.0,
        "highlight_score": 0.689,
        "event_type": "emphasized_speech",
        "gesture_tags": [
            "hand_raise",
            "pointing"
        ],
        "note": "overlap(4.70s) | motion(1.00)"
    }
]
```

---

## 🔧 Jetson Orin Nano 최적화

### 메모리 고려사항
```python
# Jetson Orin Nano 사양
# - RAM: 8GB (Ampere GPU 공유)

# 최적 설정
WHISPER_MODEL = "base"       # tiny/base 추천 (8GB 충분)
MEDIAPIPE_COMPLEXITY = 1     # 0=light, 1=full (full 권장)
POSE_DETECTION_CONF = 0.3    # 낮을수록 빠름
```

### 배포 최적화
```bash
# 1. 경량 모델 사용
python scripts/pipeline.py --video videos/lecture.mp4 --model tiny

# 2. 배치 처리 (여러 영상)
for video in videos/*.mp4; do
    python scripts/pipeline.py --video "$video"
done

# 3. 결과 수집 및 분석
python scripts/validate.py
```

---

## 🐛 문제 해결

### 오디오 파일을 찾을 수 없음
```bash
# FFmpeg 설치 확인
ffmpeg -version

# videos/ 디렉토리에 파일 배치
cp lecture.mp4 videos/
python scripts/pipeline.py --video videos/lecture.mp4
```

### Whisper 모델 다운로드 오류
```bash
# 모델 수동 다운로드
python -c "import whisper; whisper.load_model('base')"

# 캐시 위치 확인
~/.cache/whisper/               (Linux/Mac)
%USERPROFILE%\.cache\whisper\   (Windows)
```

### 메모리 부족
```bash
# 모델 크기 축소: base 대신 tiny 사용
python scripts/pipeline.py --video videos/lecture.mp4 --model tiny
```

### pose_data.json이 생성되지 않음
```bash
# 최신 pose 파일 자동 탐색 (outputs/pose_data_*.json)
# Step 0 재실행 후 Step 1 진행
python scripts/pose_analyzer.py
python scripts/extract_motion_events.py
```

---

## 📚 참고 자료

- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [NVIDIA Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/jetson-orin-nano)

---

## 📝 라이선스

This project is provided for research and development purposes.

---

## ✅ 체크리스트

시스템 설정 완료 확인:

- [ ] Python 3.10+ 설치
- [ ] 모든 의존성 설치 (`pip install -r requirements.txt`)
- [ ] FFmpeg 설치 확인 (`ffmpeg -version`)
- [ ] `videos/` 디렉토리에 테스트 영상 배치
- [ ] `python scripts/validate.py` 실행 - **READY** 상태 확인
- [ ] `python scripts/pipeline.py --video videos/test.mp4` 실행
- [ ] `outputs/highlight_segments.json` 생성 확인
- [ ] `outputs/{영상명}_highlight_reel.mp4` 생성 확인

---

**최종 업데이트**: 2026-05-16
