import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from collections import deque

st.set_page_config(page_title="Gaze Calibration", layout="centered")

st.title("🎯 Gaze Calibration")
st.markdown(
    '''
This calibration step maps your eye movement to screen coordinates.
Follow the red dots as they appear. Data will be collected for each point.
'''
)

SCREEN_W, SCREEN_H = 1280, 720
calibration_points = [
    (100, 100), (SCREEN_W // 2, 100), (SCREEN_W - 100, 100),
    (100, SCREEN_H // 2), (SCREEN_W // 2, SCREEN_H // 2), (SCREEN_W - 100, SCREEN_H // 2),
    (100, SCREEN_H - 100), (SCREEN_W // 2, SCREEN_H - 100), (SCREEN_W - 100, SCREEN_H - 100)
]

class CalibGazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.current_point = 0
        self.timer = 0
        self.collecting = False
        self.collected_data = []  # (rel_x, rel_y) -> (screen_x, screen_y)
        self.buffer = deque(maxlen=5)

    def draw_calibration_marker(self, frame, pt):
        cv2.circle(frame, pt, 20, (0, 0, 255), -1)
        return frame

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if self.current_point >= len(calibration_points):
            return img

        target = calibration_points[self.current_point]
        img = self.draw_calibration_marker(img, target)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            try:
                eye_l = face.landmark[33]
                eye_r = face.landmark[133]
                eye_width = eye_r.x - eye_l.x
                eye_height = eye_r.y - eye_l.y

                iris_pts = [face.landmark[i] for i in range(468, 478)]
                iris_x = np.mean([p.x for p in iris_pts])
                iris_y = np.mean([p.y for p in iris_pts])

                rel_x = (iris_x - eye_l.x) / eye_width
                rel_y = (iris_y - eye_l.y) / eye_height
                self.buffer.append((rel_x, rel_y))

                if len(self.buffer) == self.buffer.maxlen:
                    avg_x = np.mean([b[0] for b in self.buffer])
                    avg_y = np.mean([b[1] for b in self.buffer])
                    self.collected_data.append(((avg_x, avg_y), target))

                    self.current_point += 1
                    self.buffer.clear()
            except:
                pass

        return img

ctx = webrtc_streamer(
    key="calibration",
    video_processor_factory=CalibGazeProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

if ctx and ctx.video_processor:
    if st.button("🎓 Train Calibration Model") and ctx.video_processor.current_point >= len(calibration_points):
        X = []
        y = []
        for (rel_x, rel_y), (screen_x, screen_y) in ctx.video_processor.collected_data:
            X.append([rel_x, rel_y])
            y.append([screen_x, screen_y])
        model = LinearRegression()
        model.fit(X, y)
        with open("gaze_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("✅ Calibration complete. Model saved as gaze_model.pkl")
    elif ctx.video_processor.current_point < len(calibration_points):
        st.info("Calibration in progress... follow the red dots.")