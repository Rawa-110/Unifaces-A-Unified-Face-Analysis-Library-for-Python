# UniFace — A Unified Face Analysis Library for Python

<img width="2400" height="1122" alt="UniFace" src="https://github.com/user-attachments/assets/9d100f7b-13cf-4ecd-a279-16e2a2038c00" />

UniFace is a unified Python library that brings multiple face-analysis capabilities together in a simple and consistent framework.

It supports face detection, recognition, tracking, landmarks, demographics, emotions, gaze estimation, head pose, anti-spoofing, anonymization, and more.

---

## ✨ Features

- 👤 **Face Detection** — Detect faces in images and videos.
- 🧠 **Face Recognition** — Generate facial embeddings and identify faces.
- 🎯 **Facial Landmarks** — Detect key facial points such as the eyes, nose, and mouth.
- 😊 **Emotion Analysis** — Analyze facial expressions and emotions.
- 👨‍🦱 **Demographics** — Estimate age groups, sex, and race.
- 👁️ **Gaze Estimation** — Estimate eye gaze direction.
- 🧭 **Head Pose** — Estimate pitch, yaw, and roll.
- 🎭 **Face Parsing** — Segment facial regions and features.
- 🖼️ **Portrait Matting** — Extract people from backgrounds.
- 🛡️ **Anti-Spoofing** — Detect presentation attacks and fake faces.
- 🔒 **Anonymization** — Blur or anonymize faces using multiple methods.
- 🔎 **Vector Search** — Search facial embeddings using FAISS.
- ⚡ **Hardware Support** — Runs on CPU, Apple Silicon, and NVIDIA CUDA.

---

## 📦 Installation

### CPU / Apple Silicon

```bash
pip install "uniface[cpu]"

NVIDIA CUDA
pip install "uniface[gpu]"
Latest Pre-release
pip install --pre "uniface[cpu]"

🚀 Quick Start

UniFace provides a simple API for running multiple face-analysis tasks.

FaceAnalyzer can perform detection, alignment, and recognition in a unified pipeline.

Attribute models are opt-in, meaning you only load the models you need.

import cv2

from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(
    predictors=[FairFace()]
)

image = cv2.imread("photo.jpg")

for face in analyzer.analyze(image):
    print("Bounding Box:", face.bbox)
    print("Sex:", face.sex)
    print("Age Group:", face.age_group)
    print("Embedding Shape:", face.embedding.shape)

📌 Face Attributes

The following fields are always available:

bbox
confidence
landmarks
embedding

Additional attributes remain None until the corresponding predictor is enabled.

These may include:

age
sex
race
emotion
quality
Face state attributes

🧩 Supported Tasks & Models:

| Task                 | Supported Models                                                   |
| -------------------- | ------------------------------------------------------------------ |
| **Face Detection**   | RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace |
| **Face Recognition** | AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace                 |
| **Face Tracking**    | BYTETracker, persistent IDs across video frames                    |
| **Facial Landmarks** | 2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D)        |
| **Face Parsing**     | BiSeNet (19 classes), XSeg masking                                 |
| **Portrait Matting** | MODNet, trimap-free                                                |
| **Gaze Estimation**  | MobileGaze (ResNet-18 / 34 / 50, MobileNetV2)                      |
| **Head Pose**        | 6D rotation representation, pitch / yaw / roll                     |
| **Demographics**     | AgeGender, FairFace (age group, sex, race)                         |
| **Emotion**          | AffectNet-7, AffectNet-8                                           |
| **Face States**      | FaceAttribNet: eyes, glasses, sunglasses, mask                     |
| **Face Quality**     | eDifFIQA (T / S / M / L)                                           |
| **Anti-Spoofing**    | MiniFASNet liveness                                                |
| **Anonymization**    | 5 blur methods                                                     |
| **Vector Store**     | FAISS-backed embedding search                                      |


🧠 Analysis Pipeline

UniFace is designed around a modular pipeline where different predictors can be enabled depending on the application's requirements.

Input Image / Video
        │
        ▼
┌───────────────────┐
│   Face Detection  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Face Alignment &  │
│    Landmarks      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Face Recognition  │
│    Embeddings     │
└─────────┬─────────┘
          │
          ├──────────────► Demographics
          │
          ├──────────────► Emotion
          │
          ├──────────────► Gaze
          │
          ├──────────────► Head Pose
          │
          ├──────────────► Face States
          │
          ├──────────────► Face Quality
          │
          └──────────────► Anti-Spoofing

💻 Hardware Support

UniFace can run on multiple hardware platforms:

Platform	Support
CPU	✅
Apple Silicon	✅
NVIDIA CUDA	✅

Model weights are downloaded automatically on first use.

All downloaded weights are verified using SHA-256 checksums.

📚 Models

UniFace integrates multiple state-of-the-art and lightweight models for different face-analysis tasks.

Detection
SCRFD
CenterFace
YOLOv5-Face
YOLOv8-Face
BlazeFace
Recognition
AdaFace
ArcFace
EdgeFace
MobileFace
SphereFace
Landmarks
2d106det
PIPNet
Face Mesh
FairFace
AgeGender
AffectNet-7
AffectNet-8
Other Models
MODNet
MobileGaze
BiSeNet
FaceAttribNet
MiniFASNet
eDifFIQA

🔍 Vector Search
:

Face similarity search
Identity retrieval
Face databases
Large-scale embedding search
🎯 Use Cases

UniFace can be used in a variety of computer vision and AI applications:

🔐 Face verification
👥 Face recognition systems
📹 Video analytics
🧑‍💻 Computer vision applications
😊 Emotion analysis
👁️ Gaze tracking
🛡️ Anti-spoofing systems
🖼️ Image and video anonymization
🔎 Facial similarity search
🤖 AI-powered vision systems
⚙️ Design Philosophy

UniFace focuses on providing:

One unified API
Modular predictors
Multiple model backends
CPU, Apple Silicon, and CUDA support
Simple integration
Automatic model management
Verified model weights

The goal is to make advanced face analysis accessible through a consistent Python interface.

📝 Example:

A predictor can be enabled only when its functionality is required:

from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(
    predictors=[
        FairFace()
    ]
)

This modular approach avoids loading unnecessary attribute models.

UniFace provides FAISS-backed vector search for efficient facial embedding retrieval.

📋 API Overview
FaceAnalyzer
│
├── Detection
├── Alignment
├── Recognition
├── Landmarks
├── Demographics
├── Emotion
├── Gaze
├── Head Pose
├── Face States
├── Face Quality
├── Anti-Spoofing
├── Anonymization
└── Vector Search

Python
OpenCV
PyTorch / required backend dependencies
CPU, Apple Silicon, or NVIDIA CUDA hardware

Additional dependencies are installed automatically according to the selected installation extra.

🚧 Project Status

UniFace is actively designed as a unified framework for modern face-analysis workloads.

More models, predictors, optimizations, and integrations can be added over time.

🤝 Contributing

Contributions, ideas, bug reports, and new model integrations are welcome.

If you would like to contribute:
Create a feature branch.
Implement your changes.
Add tests where appropriate.
Open a Pull Request.
📄 License

See the LICENSE file for licensing information.

⭐ UniFace

One library. Multiple face-analysis capabilities. One unified Python API.

### أهم التغييرات التي عملتها

- رتبت الـREADME من **تعريف → Features → Installation → Quick Start → Models → Hardware → Use Cases**.
- أضفت **جدولًا واضحًا لكل المهام والموديلات**.
- فصلت **Detection / Recognition / Landmarks / Emotion / Demographics** بشكل احترافي.
- أضفت **Pipeline diagram** يعطي المشروع شكلاً أقوى على GitHub.
- صححت تنسيق كود Python والأوامر.
- أزلت عبارة `FaceAI` من العنوان والوصف حتى لا يكون هناك تضارب مع اسم المشروع **UniFace**.
- أضفت أقسام **Vector Search, API Overview, Contributing, License**.
- أبقيت رابط الصورة الذي أرسلته كما هو.

**ملاحظة مهمة:** إذا كان اسم الحزمة الفعلي الذي تريد نشره على PyPI هو `uniface`، فالأفضل أن يكون اسم المشروع في كل الـREADME هو **UniFace** وأن يبقى `uniface` فقط في أوامر `pip` وPython imports.
🧩 Supported Tasks & Models

