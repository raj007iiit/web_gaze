import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Init MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

SCREEN_W, SCREEN_H = 1280, 720

# More diverse calibration points (5x3 grid)
CALIBRATION_POINTS = [
    (int(SCREEN_W * x), int(SCREEN_H * y))
    for y in [0.2, 0.5, 0.8]
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]
]

iris_data = []
screen_data = []

cap = cv2.VideoCapture(0)
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for pt in CALIBRATION_POINTS:
    frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    cv2.circle(frame, pt, 20, (0, 0, 255), -1)
    cv2.imshow("Calibration", frame)
    time.sleep(2)

    count = 0
    while count < 30:
        ret, frame_cam = cap.read()
        frame_cam = cv2.flip(frame_cam, 1)
        rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

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

                iris_data.append([rel_x, rel_y])
                screen_data.append(pt)
                count += 1
            except:
                continue

        cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()

# Fit model with polynomial regression
X = np.array(iris_data)
y = np.array(screen_data)
model = make_pipeline(PolynomialFeatures(degree=2), Ridge())
model.fit(X, y)

with open("gaze_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Calibration complete using relative coordinates.")
