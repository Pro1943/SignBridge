# SignBridge ✋🧠

**Prototype 1 – Real-Time Sign & Gesture Recognition**

SignBridge is an AI-powered assistive communication system designed to bridge the gap between sign language users and non-signers. This repository contains **Prototype 1**, which focuses on **real-time hand tracking and basic sign/gesture interpretation** using computer vision.

> 🚧 This is an early-stage prototype built for learning, experimentation, and proof-of-concept.

---

## 🌍 Problem Statement

In classrooms and everyday interactions, deaf or hard-of-hearing individuals often face communication barriers due to the lack of accessible sign language interpretation tools. Human interpreters are not always available, scalable, or affordable.

---

## 💡 Solution Overview

SignBridge aims to provide a **real-time, camera-based sign language assistant** that:
- Detects hand landmarks using AI
- Interprets basic sign language gestures
- Translates gestures into understandable text
- Lays the foundation for two-way communication in the future

Prototype 1 focuses purely on **gesture recognition**, not full language translation.

---

## ✨ Features (Prototype 1)

- 📷 Real-time webcam input
- ✋ Accurate hand landmark detection (MediaPipe Task API)
- 🧠 Static gesture recognition:
  - 👍 Thumbs Up → **GOOD / OK / FINE**
  - 👎 Thumbs Down → **NOT GOOD / NOT OK**
  - ✊ All fingers closed → **Letter "A" (ASL)**
- 👋 Dynamic gesture recognition:
  - Hand wave → **HELLO**
- ⚡ Real-time FPS display
- 🛑 Gesture cooldown to prevent repeated triggers

---

## 🛠️ Tech Stack

- **Python 3.12**
- **OpenCV** – video capture & rendering
- **MediaPipe Tasks API** – hand landmark detection
- **NumPy** – numerical operations

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

- Python 3.12
- A working webcam

### 2️⃣ Install dependencies:
```bash
pip install opencv-python mediapipe numpy
```

### 3️⃣ Run the Prototype

```bash
python signbridge.py
```

Press **Q** to quit.

---

## 🧪 Current Limitations

- Supports only one hand
- Recognizes a small set of gestures
- Rule-based logic (no ML classification yet)
- No speech output or reverse translation (text → sign)

These limitations are intentional for Prototype 1.

---

## 🛣️ Roadmap

- 🔊 Text-to-Speech output
- 🧠 ML-based gesture classifier
- 🔤 Expanded sign vocabulary
- 🔄 Two-way communication (text/speech → sign)
- 📱 Mobile & web deployment

---

## 🎓 Learning Goals

This project is also a personal learning journey toward becoming an **ML Engineer**, covering:
- Computer Vision fundamentals
- Real-time AI systems
- Gesture analysis
- Scalable project architecture

---

## ⚠️ Disclaimer

SignBridge is **not a certified medical or accessibility device**. It is an experimental educational project and should not replace professional sign language interpreters.

---

## 📜 License

This project is licensed under the Creative Commons Attribution–NonCommercial 4.0
International License (CC BY-NC 4.0).

Commercial use is strictly prohibited.
Attribution to the original author (Pro 1943) is mandatory.

---

## 👤 Author

**Pro 1943**  
Student | Aspiring ML Engineer  

---

> "Building never stops." 🚀
