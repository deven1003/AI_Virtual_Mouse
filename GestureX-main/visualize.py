# MINI PROJECT
# HAND GESTURE RECOGNITION

from tensorflow.keras.models import load_model
from ann_visualizer.visualize import ann_viz


model = load_model("model_1.h5")

ann_viz(model, view=True, title="Model Architecture")
