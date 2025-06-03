import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Load gaze data and background image
df = pd.read_csv("gaze_data.csv")
bg = cv2.imread("stimulus.jpg")
bg = cv2.resize(bg, (1280, 720))
bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12.8, 7.2))
plt.imshow(bg_rgb)

# ✅ FIX: Use colormap object directly
cmap = plt.get_cmap("jet")

sns.kdeplot(
    x=df["x"],
    y=df["y"],
    cmap=cmap,      # Pass actual colormap, not string
    fill=True,
    alpha=0.5,
    bw_adjust=0.2,
    thresh=0.05
)

plt.axis('off')
plt.tight_layout()
plt.savefig("gaze_heatmap.png")
plt.show()
