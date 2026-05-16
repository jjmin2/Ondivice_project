import json
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parent.parent

JSON_PATH = BASE_DIR / "outputs" / "pose_data_20260516_145846.json"

# JSON 불러오기
with open(JSON_PATH, "r", encoding="utf-8") as f:
    pose_data = json.load(f)

timestamps = []
movement_scores = []

# 데이터 추출
for frame in pose_data:

    timestamps.append(frame["timestamp"])
    movement_scores.append(frame["movement_score"])

# 그래프 출력
plt.figure(figsize=(12, 5))

plt.plot(timestamps, movement_scores)

plt.title("Movement Score Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Movement Score")

plt.grid(True)

plt.show()