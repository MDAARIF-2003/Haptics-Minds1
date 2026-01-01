import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import uuid

# -------- LOAD MODEL --------
MODEL_PATH = "pattern_model_v2.h5"

try:
    model = load_model(MODEL_PATH)
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

PATTERN_CLASSES = ['CHECKS', 'DOTS', 'FLORAL', 'PLAINS', 'STRIPES']
IMG_SIZE = 64

last_result_text = ""


# -------- TEXT → AUDIO --------
def text_to_speech(text):
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename


# -------- CORE LOGIC --------
def analyze_image(image):
    global last_result_text

    if image is None:
        return "Please upload an image.", None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    if not MODEL_LOADED:
        last_result_text = "Model not found. Please train the model."
        audio = text_to_speech(last_result_text)
        return last_result_text, audio

    preds = model.predict(roi, verbose=0)
    pattern = PATTERN_CLASSES[np.argmax(preds)]

    last_result_text = f"Detected clothing pattern is {pattern}"
    audio = text_to_speech(last_result_text)

    return last_result_text, audio


# -------- REPEAT VOICE --------
def repeat_voice():
    if last_result_text == "":
        return "No detection yet.", None

    audio = text_to_speech(last_result_text)
    return last_result_text, audio


# -------- FRONTEND --------
with gr.Blocks() as demo:
    gr.Markdown("## 👁️‍🗨️ Vision Beyond Sight")
    gr.Markdown("AI-powered clothing pattern recognition with voice assistance")

    image_input = gr.Image(type="numpy", label="Upload Clothing Image")
    output_text = gr.Textbox(label="Detection Result")
    audio_output = gr.Audio(label="Voice Output", autoplay=True)

    detect_btn = gr.Button("Detect Pattern")
    repeat_btn = gr.Button("🔁 Repeat Voice")

    detect_btn.click(
        analyze_image,
        inputs=image_input,
        outputs=[output_text, audio_output]
    )

    repeat_btn.click(
        repeat_voice,
        outputs=[output_text, audio_output]
    )

demo.launch(share=True)
