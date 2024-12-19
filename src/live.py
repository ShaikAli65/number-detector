import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Load the model if available
# Load the model if available
model_path = 'my_model.keras'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
else:
    print(f"Model file '{model_path}' not found.")
    exit()

