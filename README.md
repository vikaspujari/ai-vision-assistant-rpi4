# Smart AI Vision Assistant — Raspberry Pi 4B

A real-time object detection and voice interaction system 
running fully on-device (offline speech) with cloud-augmented 
AI responses via Gemini.

## 🔧 Hardware
- Raspberry Pi 4B
- Pi Camera Module
- USB Microphone
- Speaker (3.5mm / USB)

## 🧠 What It Does
- Detects objects in real time using YOLOv4-tiny (OpenCV DNN)
- Classifies position as left / front / right relative to frame
- Accepts offline voice queries via VOSK speech recognition
- Answers questions about the scene using Google Gemini 1.5 Flash
- Responds via Piper TTS — fully spoken audio output

## 🗂️ Tech Stack
| Component        | Tool/Library              |
|------------------|---------------------------|
| Object Detection | YOLOv4-tiny + OpenCV DNN  |
| Speech Input     | VOSK (offline, on-device) |
| AI Response      | Google Gemini 1.5 Flash   |
| Text-to-Speech   | Piper TTS                 |
| Camera           | Picamera2                 |
| Threading        | Python threading + queue  |

## 📊 Performance (Pi 4B)
- Detection: ~10–12 FPS at 320×320 input
- Voice latency: <1s (VOSK offline recognition)
- Detects: person, car, bicycle, motorbike, bus, truck

## 🚀 Setup
```bash
pip install opencv-python vosk sounddevice \
    google-generativeai picamera2
