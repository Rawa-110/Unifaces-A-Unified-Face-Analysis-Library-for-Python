# Unifaces-A-Unified-Face-Analysis-Library-for-Python

<img width="2400" height="1122" alt="image" src="https://github.com/user-attachments/assets/9d100f7b-13cf-4ecd-a279-16e2a2038c00" />

FaceAI: A Unified Face Analysis Library for Python

Description:
FaceAI is a Python library designed to bring multiple face-analysis capabilities together in one simple and unified framework.

👤 Face Detection — Detect faces in images and videos.
📍 Facial Landmarks — Identify key points such as the eyes, nose, and mouth.
😊 Expression Analysis — Analyze facial expressions and emotions.
🔍 Face Feature Analysis — Extract useful facial characteristics.
🐍 Python-Friendly API — Easy to integrate into Python projects.
⚡ Computer Vision Applications — Suitable for AI, computer vision, and face-analysis projects.

Suggested image concept:
A person facing a camera with a digital facial-analysis overlay, showing landmark points around the eyes, nose, mouth, and face contour. This would communicate AI-powered face analysis more clearly than the original character.

pip install "uniface[cpu]"          # CPU and Apple Silicon
pip install "uniface[gpu]"          # NVIDIA CUDA
pip install --pre "uniface[cpu]"    # latest pre-release

A first script:

FaceAnalyzer runs detection, alignment and recognition in one call. Attribute models are opt-in.

import cv2
from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(predictors=[FairFace()])

for face in analyzer.analyze(cv2.imread("photo.jpg")):
    print(face.bbox, face.sex, face.age_group, face.embedding.shape)

    bbox, confidence, landmarks and embedding are always set. Age, sex, race, emotion, quality and the face states stay None until you pass the predictor that fills them.

    All fifteen tasks, and which model does each

Task	Models
Face Detection	RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace
Face Recognition	AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace
Face Tracking	BYTETracker, persistent IDs across video frames
Facial Landmarks	2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D)
Face Parsing	BiSeNet (19 classes), XSeg masking
Portrait Matting	MODNet, trimap-free
Gaze Estimation	MobileGaze (ResNet-18 / 34 / 50, MobileNetV2)
Head Pose	6D rotation representation, pitch / yaw / roll
Demographics	AgeGender, FairFace (age group, sex, race)
Emotion	AffectNet-7 and AffectNet-8
Face States	FaceAttribNet: eyes, glasses, sunglasses, mask
Face Quality	eDifFIQA (T / S / M / L)
Anti-Spoofing	MiniFASNet liveness
Anonymization	5 blur methods
Vector Store	FAISS-backed embedding search
Runs on CPU, Apple Silicon and CUDA. Weights download on first use, verified by SHA-256.



    
