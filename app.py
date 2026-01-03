import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
import uuid

# ---------------- CONFIG ----------------
MODEL_PATH = "pattern_model_v2.h5"
IMG_SIZE = 64

# ⚠️ ORDER MUST MATCH TRAINING FOLDERS
INDEX_TO_PATTERN = {
    0: "CHECKS",
    1: "DOTS",
    2: "FLORAL",
    3: "PLAINS",
    4: "STRIPES"
}

last_result_text = ""

# ---------------- LOAD MODEL ----------------
try:
    model = load_model(MODEL_PATH, compile=False)
    MODEL_LOADED = True
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    MODEL_LOADED = False

# ---------------- TEXT → SPEECH ----------------
def text_to_speech(text):
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# ---------------- CORE LOGIC ----------------
def analyze_image(image):
    global last_result_text

    if image is None:
        return "Please upload an image.", None

    if not MODEL_LOADED:
        last_result_text = "AI model not loaded. Please upload the trained model."
        return last_result_text, text_to_speech(last_result_text)

    # Convert image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict
    preds = model.predict(roi, verbose=0)[0]
    class_id = int(np.argmax(preds))
    pattern = INDEX_TO_PATTERN.get(class_id, "UNKNOWN")

    last_result_text = f"Detected clothing pattern is {pattern}"
    audio = text_to_speech(last_result_text)

    return last_result_text, audio

# ---------------- REPEAT VOICE ----------------
def repeat_voice():
    if last_result_text == "":
        return "No detection yet.", None

    return last_result_text, text_to_speech(last_result_text)

# ---------------- UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 👁️‍🗨️ Vision Beyond Sight")
    gr.Markdown("AI-powered **clothing pattern recognition** with voice assistance")

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

demo.launch()
