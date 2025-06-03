import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import csv
import os
from collections import deque

stimulus_path = "stimulus_input"
model_path = "gaze_model.pkl"
output_file = "gaze_data.csv"

img_exts = [".jpg", ".jpeg", ".png"]
vid_exts = [".mp4", ".avi"]
stimulus_file = None

# Find stimulus
for ext in img_exts + vid_exts:
    if os.path.exists(stimulus_path + ext):
        stimulus_file = stimulus_path + ext
        break

if not stimulus_file:
    raise FileNotFoundError("❌ No valid stimulus file found.")

is_video = os.path.splitext(stimulus_file)[-1].lower() in vid_exts
SCREEN_W, SCREEN_H = 1280, 720

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)
gaze_buffer = deque(maxlen=5)

# Save CSV in current directory
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp_or_frame", "x", "y"])

    # Set up display
    cv2.namedWindow("Gaze Stimulus", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Stimulus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if is_video:
        video = cv2.VideoCapture(stimulus_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        while True:
            ret_vid, frame_vid = video.read()
            ret_cam, frame_cam = cap.read()
            if not ret_vid or not ret_cam:
                break

            frame_vid = cv2.resize(frame_vid, (SCREEN_W, SCREEN_H))
            frame_cam = cv2.flip(frame_cam, 1)
            rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                try:
                    face = results.multi_face_landmarks[0]
                    eye_l, eye_r = face.landmark[33], face.landmark[133]
                    eye_width = eye_r.x - eye_l.x
                    eye_height = eye_r.y - eye_l.y
                    iris_pts = [face.landmark[i] for i in range(468, 478)]
                    iris_x = np.mean([p.x for p in iris_pts])
                    iris_y = np.mean([p.y for p in iris_pts])
                    rel_x = (iris_x - eye_l.x) / eye_width
                    rel_y = (iris_y - eye_l.y) / eye_height

                    pred_x, pred_y = model.predict([[rel_x, rel_y]])[0]
                    pred_x = int(np.clip(pred_x, 0, SCREEN_W - 1))
                    pred_y = int(np.clip(pred_y, 0, SCREEN_H - 1))

                    gaze_buffer.append((pred_x, pred_y))
                    smooth_x = int(np.mean([g[0] for g in gaze_buffer]))
                    smooth_y = int(np.mean([g[1] for g in gaze_buffer]))

                    cv2.circle(frame_vid, (smooth_x, smooth_y), 15, (0, 0, 255), -1)
                    writer.writerow([frame_idx, smooth_x, smooth_y])
                except:
                    pass

            cv2.imshow("Gaze Stimulus", frame_vid)
            frame_idx += 1
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        video.release()

    else:
        # Static image mode
        stimulus = cv2.imread(stimulus_file)
        stimulus = cv2.resize(stimulus, (SCREEN_W, SCREEN_H))
        print("⚡ Showing image for 20 seconds...")

        start_time = time.time()
        while time.time() - start_time < 20:
            ret_cam, frame_cam = cap.read()
            if not ret_cam:
                break

            frame_cam = cv2.flip(frame_cam, 1)
            rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            overlay = stimulus.copy()
            if results.multi_face_landmarks:
                try:
                    face = results.multi_face_landmarks[0]
                    eye_l, eye_r = face.landmark[33], face.landmark[133]
                    eye_width = eye_r.x - eye_l.x
                    eye_height = eye_r.y - eye_l.y
                    iris_pts = [face.landmark[i] for i in range(468, 478)]
                    iris_x = np.mean([p.x for p in iris_pts])
                    iris_y = np.mean([p.y for p in iris_pts])
                    rel_x = (iris_x - eye_l.x) / eye_width
                    rel_y = (iris_y - eye_l.y) / eye_height

                    pred_x, pred_y = model.predict([[rel_x, rel_y]])[0]
                    pred_x = int(np.clip(pred_x, 0, SCREEN_W - 1))
                    pred_y = int(np.clip(pred_y, 0, SCREEN_H - 1))

                    gaze_buffer.append((pred_x, pred_y))
                    smooth_x = int(np.mean([g[0] for g in gaze_buffer]))
                    smooth_y = int(np.mean([g[1] for g in gaze_buffer]))

                    cv2.circle(overlay, (smooth_x, smooth_y), 15, (0, 0, 255), -1)
                    writer.writerow([time.time(), smooth_x, smooth_y])
                except:
                    pass

            cv2.imshow("Gaze Stimulus", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
print("✅ Gaze tracking completed.")
