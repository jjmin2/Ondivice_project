import cv2
import mediapipe as mp
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# 명령줄 인자 처리
if len(sys.argv) > 1:
    video_path = sys.argv[1]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_JSON = BASE_DIR / "outputs" / f"pose_data_{timestamp_str}.json"
else:
    # 웹캠 모드 (기본)
    video_path = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_JSON = BASE_DIR / "outputs" / f"pose_data_{timestamp_str}.json"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

# 비디오 파일 또는 웹캠 사용
cap = cv2.VideoCapture(video_path)

pose_data = []
prev_landmarks = None

KEYPOINTS = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "nose": 0
}

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # 프레임 크기 고정
    frame = cv2.resize(frame, (640, 480))

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    frame_info = {
        "timestamp": timestamp,
        "movement_score": 0,
        "keypoints": {}
    }

    if results.pose_landmarks:

        current_landmarks = {}

        for name, idx in KEYPOINTS.items():

            lm = results.pose_landmarks.landmark[idx]

            current_landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            }

        frame_info["keypoints"] = current_landmarks

        # movement score 계산
        movement_score = 0

        if prev_landmarks is not None:

            for name in KEYPOINTS.keys():

                prev = prev_landmarks[name]
                curr = current_landmarks[name]

                dist = np.sqrt(
                    (curr["x"] - prev["x"]) ** 2 +
                    (curr["y"] - prev["y"]) ** 2
                )

                movement_score += dist

        frame_info["movement_score"] = movement_score

        prev_landmarks = current_landmarks

        # landmark 시각화
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # movement score 표시
        cv2.putText(
            frame,
            f"Movement: {movement_score:.4f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    pose_data.append(frame_info)

    cv2.imshow("Pose Detection", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# JSON 저장
with open(str(OUTPUT_JSON), "w", encoding="utf-8") as f:
    print(len(pose_data))
    json.dump(pose_data, f, indent=4)

print("JSON SAVED")
print(f"Pose data saved to {OUTPUT_JSON}")