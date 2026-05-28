
# =========================================================
# GEMINI API KEY
# =========================================================
API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"

# =========================================================
# IMPORTS
# =========================================================
import cv2
import numpy as np
import os
import queue
import json
import threading
import sounddevice as sd
import google.generativeai as genai

from vosk import Model, KaldiRecognizer
from picamera2 import Picamera2

# =========================================================
# GEMINI SETUP
# =========================================================
genai.configure(api_key=API_KEY)

ai_model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# =========================================================
# CONFIG
# =========================================================
YOLO_CFG = "/home/pi4/yolov4-tiny.cfg"
YOLO_WEIGHTS = "/home/pi4/yolov4-tiny.weights"
COCO_NAMES = "/home/pi4/coco.names"

VOSK_MODEL = "/home/pi4/model"

PIPER_MODEL = "/home/pi4/en_US-lessac-medium.onnx"
PIPER_CONFIG = "/home/pi4/en_US-lessac-medium.onnx.json"

MIC_RATE = 16000

# =========================================================
# SPEAK
# =========================================================
audio_queue = queue.Queue()

def audio_worker():
    while True:

        text = audio_queue.get()

        if text is None:
            break

        safe = text.replace('"', '').replace("'", "")

        print("Bot:", text)

        os.system(
            f'echo "{safe}" | piper '
            f'--model {PIPER_MODEL} '
            f'--config {PIPER_CONFIG} '
            f'--output_raw 2>/dev/null | '
            f'aplay -f S16_LE -r 22050'
        )

        audio_queue.task_done()

threading.Thread(
    target=audio_worker,
    daemon=True
).start()

def speak(text):
    audio_queue.put(text)

# =========================================================
# LOAD YOLO
# =========================================================
print("Loading YOLO...")

net = cv2.dnn.readNet(
    YOLO_WEIGHTS,
    YOLO_CFG
)

net.setPreferableBackend(
    cv2.dnn.DNN_BACKEND_OPENCV
)

net.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CPU
)

with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

allowed = [
    "person",
    "car",
    "bicycle",
    "motorbike",
    "bus",
    "truck"
]

# =========================================================
# PICAMERA2
# =========================================================
print("Starting Camera...")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)

picam2.start()

# =========================================================
# VOSK
# =========================================================
print("Loading VOSK...")

model = Model(VOSK_MODEL)

recognizer = KaldiRecognizer(
    model,
    MIC_RATE
)

q = queue.Queue()

def callback(indata, frames, time_info, status):

    if status:
        return

    q.put(bytes(indata))

# =========================================================
# DETECTED OBJECTS
# =========================================================
detected_objects = []

# =========================================================
# GEMINI CHATBOT
# =========================================================
def ask_gemini(query):

    try:

        scene = ", ".join(detected_objects)

        prompt = f"""
        You are a smart Raspberry Pi vision assistant.

        Objects detected:
        {scene}

        User question:
        {query}

        Give a short voice-friendly answer.
        """

        response = ai_model.generate_content(
            prompt
        )

        answer = response.text.strip()

        return answer[:200]

    except Exception as e:

        print("Gemini Error:", e)

        return "I could not connect to Gemini"

# =========================================================
# MAIN
# =========================================================
print("System Ready")

speak("System ready")

with sd.RawInputStream(
    samplerate=MIC_RATE,
    blocksize=8000,
    dtype='int16',
    channels=1,
    callback=callback
):

    while True:

        # =================================================
        # CAMERA
        # =================================================
        frame = picam2.capture_array()

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )

        height, width = frame.shape[:2]

        # =================================================
        # YOLO
        # =================================================
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255,
            (320, 320),
            swapRB=True
        )

        net.setInput(blob)

        outputs = net.forward(layer_names)

        detected_objects.clear()

        for output in outputs:

            for detection in output:

                scores = detection[5:]

                class_id = np.argmax(scores)

                confidence = scores[class_id]

                if confidence > 0.5:

                    label_name = classes[class_id]

                    if label_name not in allowed:
                        continue

                    center_x = int(
                        detection[0] * width
                    )

                    center_y = int(
                        detection[1] * height
                    )

                    w = int(
                        detection[2] * width
                    )

                    h = int(
                        detection[3] * height
                    )

                    x = int(center_x - w / 2)

                    y = int(center_y - h / 2)

                    # =====================================
                    # POSITION
                    # =====================================
                    if center_x < width // 3:
                        pos = "left"

                    elif center_x > 2 * width // 3:
                        pos = "right"

                    else:
                        pos = "front"

                    label = f"{label_name} {pos}"

                    detected_objects.append(label)

                    # =====================================
                    # DRAW
                    # =====================================
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        # =================================================
        # VOICE INPUT
        # =================================================
        if not q.empty():

            data = q.get()

            if recognizer.AcceptWaveform(data):

                result = json.loads(
                    recognizer.Result()
                )

                query = result.get(
                    "text",
                    ""
                )

                if query:

                    print("You:", query)

                    if "exit" in query:
                        speak("Goodbye")
                        break

                    answer = ask_gemini(query)

                    speak(answer)

        # =================================================
        # DISPLAY
        # =================================================
        cv2.imshow(
            "Smart AI Assistant",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

# =========================================================
# CLEANUP
# =========================================================
picam2.stop()

cv2.destroyAllWindows()

audio_queue.put(None)

print("System Closed")
