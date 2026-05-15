import cv2
import mediapipe as mp
import json
import numpy as np
import math
from pathlib import Path
from datetime import datetime
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# Gesture threshold 설정
WIDE_GESTURE_THRESHOLD = 0.35  # 좌우 손 거리 임계값
POINTING_ANGLE_THRESHOLD = 150.0  # 팔이 거의 직선이면 pointing
MOTION_THRESHOLD = 0.05  # gesture 감지를 위한 최소 움직임
GESTURE_MIN_FRAMES = 3  # gesture로 인정할 최소 연속 프레임 수
WRIST_VELOCITY_THRESHOLD = 0.01  # 손목 속도 임계값

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


class GestureDetector:
    """Motion-aware gesture detection with temporal consistency"""

    def __init__(self):
        self.prev_landmarks = None
        self.gesture_states = {
            "hand_raise": {"active": False, "count": 0},
            "wide_gesture": {"active": False, "count": 0},
            "pointing": {"active": False, "count": 0}
        }

    def calculate_velocity(self, current: dict, prev: dict) -> float:
        """두 프레임 간 속도 계산"""
        if not prev:
            return 0.0
        dx = current["x"] - prev["x"]
        dy = current["y"] - prev["y"]
        return math.sqrt(dx**2 + dy**2)

    def detect_hand_raise(self, landmarks: dict, movement_score: float) -> bool:
        """손을 드는 움직임 감지"""
        left_wrist = landmarks.get("left_wrist")
        right_wrist = landmarks.get("right_wrist")
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")

        # 기본 조건: 손목이 어깨보다 위
        left_raise = left_wrist and left_shoulder and left_wrist["y"] < left_shoulder["y"]
        right_raise = right_wrist and right_shoulder and right_wrist["y"] < right_shoulder["y"]

        if not (left_raise or right_raise):
            return False

        # 움직임 조건
        if movement_score < MOTION_THRESHOLD:
            return False

        # 속도 조건 (손목이 위로 이동 중)
        if self.prev_landmarks:
            left_vel = self.calculate_velocity(left_wrist, self.prev_landmarks.get("left_wrist")) if left_wrist else 0
            right_vel = self.calculate_velocity(right_wrist, self.prev_landmarks.get("right_wrist")) if right_wrist else 0

            # 손목이 위로 이동하는 속도가 임계값 이상
            wrist_moving_up = False
            if left_wrist and self.prev_landmarks.get("left_wrist"):
                if left_wrist["y"] < self.prev_landmarks["left_wrist"]["y"]:  # y 감소 = 위로 이동
                    wrist_moving_up = True
            if right_wrist and self.prev_landmarks.get("right_wrist"):
                if right_wrist["y"] < self.prev_landmarks["right_wrist"]["y"]:
                    wrist_moving_up = True

            if not wrist_moving_up:
                return False

        return True

    def detect_wide_gesture(self, landmarks: dict, movement_score: float) -> bool:
        """팔을 벌리는 움직임 감지"""
        left_wrist = landmarks.get("left_wrist")
        right_wrist = landmarks.get("right_wrist")

        if not (left_wrist and right_wrist):
            return False

        # 기본 조건: 손 사이 거리가 임계값 이상
        hand_distance = abs(left_wrist["x"] - right_wrist["x"])
        if hand_distance <= WIDE_GESTURE_THRESHOLD:
            return False

        # 움직임 조건
        if movement_score < MOTION_THRESHOLD:
            return False

        # 팔 벌리는 방향으로 이동 중인지 확인
        if self.prev_landmarks:
            prev_left = self.prev_landmarks.get("left_wrist")
            prev_right = self.prev_landmarks.get("right_wrist")

            if prev_left and prev_right:
                prev_distance = abs(prev_left["x"] - prev_right["x"])
                # 현재 거리가 이전보다 커지는 중
                if hand_distance <= prev_distance:
                    return False

                # 양 손이 바깥으로 이동 중
                left_moving_out = left_wrist["x"] < prev_left["x"]  # 왼쪽 손 왼쪽으로
                right_moving_out = right_wrist["x"] > prev_right["x"]  # 오른쪽 손 오른쪽으로

                if not (left_moving_out or right_moving_out):
                    return False

        return True

    def detect_pointing(self, landmarks: dict, movement_score: float) -> bool:
        """팔을 펴는 움직임 감지"""
        left_shoulder = landmarks.get("left_shoulder")
        left_elbow = landmarks.get("left_elbow")
        left_wrist = landmarks.get("left_wrist")
        right_shoulder = landmarks.get("right_shoulder")
        right_elbow = landmarks.get("right_elbow")
        right_wrist = landmarks.get("right_wrist")

        pointing_left = False
        pointing_right = False

        # 왼팔 pointing 체크
        if left_shoulder and left_elbow and left_wrist:
            angle = calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)
            if angle > POINTING_ANGLE_THRESHOLD:
                pointing_left = True

        # 오른팔 pointing 체크
        if right_shoulder and right_elbow and right_wrist:
            angle = calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
            if angle > POINTING_ANGLE_THRESHOLD:
                pointing_right = True

        if not (pointing_left or pointing_right):
            return False

        # 움직임 조건
        if movement_score < MOTION_THRESHOLD:
            return False

        # 손목 속도 조건
        wrist_velocity = 0
        if self.prev_landmarks:
            if pointing_left and left_wrist and self.prev_landmarks.get("left_wrist"):
                wrist_velocity = max(wrist_velocity, self.calculate_velocity(left_wrist, self.prev_landmarks["left_wrist"]))
            if pointing_right and right_wrist and self.prev_landmarks.get("right_wrist"):
                wrist_velocity = max(wrist_velocity, self.calculate_velocity(right_wrist, self.prev_landmarks["right_wrist"]))

        if wrist_velocity < WRIST_VELOCITY_THRESHOLD:
            return False

        return True

    def detect_gesture(self, landmarks: dict, movement_score: float) -> list[str]:
        """통합 gesture detection with temporal consistency"""
        active_gestures = []

        # 각 gesture 감지
        gestures_to_check = {
            "hand_raise": self.detect_hand_raise(landmarks, movement_score),
            "wide_gesture": self.detect_wide_gesture(landmarks, movement_score),
            "pointing": self.detect_pointing(landmarks, movement_score)
        }

        for gesture_name, detected in gestures_to_check.items():
            state = self.gesture_states[gesture_name]

            if detected:
                state["count"] += 1
                # 최소 프레임 수 이상 연속 감지되면 active
                if state["count"] >= GESTURE_MIN_FRAMES:
                    if not state["active"]:
                        state["active"] = True
                        active_gestures.append(gesture_name)
                        print(f"[Gesture START] {gesture_name} (frames: {state['count']}, movement: {movement_score:.3f})")
                    else:
                        active_gestures.append(gesture_name)
            else:
                if state["active"]:
                    print(f"[Gesture END] {gesture_name} (duration: {state['count']} frames)")
                state["active"] = False
                state["count"] = 0

        # 이전 landmark 저장
        self.prev_landmarks = landmarks.copy()

        return active_gestures


def calculate_elbow_angle(shoulder: dict, elbow: dict, wrist: dict) -> float:
    """Shoulder-elbow-wrist 각도 계산"""
    a = np.array([shoulder["x"] - elbow["x"], shoulder["y"] - elbow["y"]])
    b = np.array([wrist["x"] - elbow["x"], wrist["y"] - elbow["y"]])
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_value = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_value)))


# Gesture detector 인스턴스 생성
gesture_detector = GestureDetector()


def detect_gesture(landmarks: dict, wide_threshold: float = WIDE_GESTURE_THRESHOLD) -> list[str]:
    """Legacy function for backward compatibility"""
    return gesture_detector.detect_gesture(landmarks, 0.0)  # movement_score는 별도 계산


# 비디오 파일 또는 웹캠 사용
cap = cv2.VideoCapture(video_path)

pose_data = []
prev_landmarks = None

KEYPOINTS = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
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
        "keypoints": {},
        "gesture_tags": []
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

        gesture_tags = gesture_detector.detect_gesture(current_landmarks, movement_score)
        frame_info["gesture_tags"] = gesture_tags
        if gesture_tags:
            print(f"[Gesture] {timestamp:.2f}s -> {gesture_tags}")

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