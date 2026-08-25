🧑‍💻 Face Recognition & Landmarks

UniFace provides a complete face analysis pipeline, including face detection, facial landmarks, face alignment, face recognition, embeddings, face verification, and face search.

🔍 Face Recognition

Face recognition models extract a numerical representation of a face called an embedding. These embeddings can be used to compare faces, verify identities, or search for a person in a database.

Model	Backbone	Size	Embedding Dim
🟣 AdaFace	IR-18 / IR-101	92–249 MB	512
🔵 ArcFace	MobileNet / ResNet	8–166 MB	512
🟢 EdgeFace	EdgeNeXt / LoRA	5–70 MB	512
🟡 MobileFace	MobileNet V2 / V3	1–10 MB	512
🔴 SphereFace	Sphere20 / 36	50–92 MB	512
🟣 AdaFace

Face recognition using an adaptive margin based on image quality.

Basic Usage
from uniface.detection import RetinaFace
from uniface.recognition import AdaFace

detector = RetinaFace()
recognizer = AdaFace()

faces = detector.detect(image)

if faces:
    embedding = recognizer.get_normalized_embedding(
        image,
        faces[0].landmarks
    )

    print(f"Embedding shape: {embedding.shape}")  # (512,)
Model Variants
from uniface.constants import AdaFaceWeights

# Lightweight
recognizer = AdaFace(
    model_name=AdaFaceWeights.IR_18
)

# High accuracy
recognizer = AdaFace(
    model_name=AdaFaceWeights.IR_101
)

# CPU execution
recognizer = AdaFace(
    providers=['CPUExecutionProvider']
)
Variant	Dataset	Size	IJB-B	IJB-C
IR_18	WebFace4M	92 MB	93.03%	94.99%
IR_101	WebFace12M	249 MB	—	97.66%
🔵 ArcFace

Face recognition using Additive Angular Margin Loss.

Basic Usage
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace

detector = RetinaFace()
recognizer = ArcFace()

faces = detector.detect(image)

if faces:
    embedding = recognizer.get_normalized_embedding(
        image,
        faces[0].landmarks
    )

    print(f"Embedding shape: {embedding.shape}")  # (512,)
Model Variants
from uniface.constants import ArcFaceWeights

# Lightweight
recognizer = ArcFace(
    model_name=ArcFaceWeights.MNET
)

# High accuracy
recognizer = ArcFace(
    model_name=ArcFaceWeights.RESNET
)

# CPU execution
recognizer = ArcFace(
    providers=['CPUExecutionProvider']
)
Variant	Backbone	Size	LFW	CFP-FP	AgeDB-30	IJB-C
MNET	MobileNet	8 MB	99.70%	98.00%	96.58%	95.02%
RESNET	ResNet50	166 MB	99.83%	99.33%	98.23%	97.25%
🟢 EdgeFace

An efficient face recognition model designed for edge devices, using an EdgeNeXt backbone with optional LoRA compression.

from uniface.detection import RetinaFace
from uniface.recognition import EdgeFace

detector = RetinaFace()
recognizer = EdgeFace()

faces = detector.detect(image)

if faces:
    embedding = recognizer.get_normalized_embedding(
        image,
        faces[0].landmarks
    )

    print(f"Embedding shape: {embedding.shape}")  # (512,)
Model Variants
from uniface.constants import EdgeFaceWeights

# Ultra-compact
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.XXS
)

# Compact
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.XS_GAMMA_06
)

# Small
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.S_GAMMA_05
)

# Full-size
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.BASE
)
Variant	Params	MFLOPs	Size	LFW	CALFW	CPLFW
XXS	1.24M	94	~5 MB	99.57%	94.83%	90.27%
XS_GAMMA_06	1.77M	154	~7 MB	99.73%	95.28%	91.58%
S_GAMMA_05	3.65M	306	~14 MB	99.78%	95.55%	92.48%
BASE	18.2M	1399	~70 MB	99.83%	96.07%	93.75%
🟡 MobileFace

Lightweight face recognition models based on MobileNet, suitable for applications where model size and inference speed are important.

from uniface.recognition import MobileFace

recognizer = MobileFace()

embedding = recognizer.get_normalized_embedding(
    image,
    landmarks
)
Model Variants
from uniface.constants import MobileFaceWeights

# Ultra-lightweight
recognizer = MobileFace(
    model_name=MobileFaceWeights.MNET_025
)

# Balanced
recognizer = MobileFace(
    model_name=MobileFaceWeights.MNET_V2
)

# Higher accuracy
recognizer = MobileFace(
    model_name=MobileFaceWeights.MNET_V3_LARGE
)
Variant	Params	Size	LFW	CALFW	CPLFW	AgeDB-30
MNET_025	0.36M	1 MB	98.76%	92.02%	82.37%	90.02%
MNET_V2	2.29M	4 MB	99.55%	94.87%	86.89%	95.16%
MNET_V3_SMALL	1.25M	3 MB	99.30%	93.77%	85.29%	92.79%
MNET_V3_LARGE	3.52M	10 MB	99.53%	94.56%	86.79%	95.13%
🔴 SphereFace

Face recognition using Angular Softmax Loss (A-Softmax).

from uniface.recognition import SphereFace
from uniface.constants import SphereFaceWeights

recognizer = SphereFace(
    model_name=SphereFaceWeights.SPHERE20
)

embedding = recognizer.get_normalized_embedding(
    image,
    landmarks
)
Variant	Params	Size	LFW	CALFW	CPLFW	AgeDB-30
SPHERE20	24.5M	50 MB	99.67%	95.61%	88.75%	96.58%
SPHERE36	34.6M	92 MB	99.72%	95.64%	89.92%	96.83%
🧮 Face Comparison

After extracting embeddings, you can calculate the similarity between two faces.

from uniface.face_utils import compute_similarity
import numpy as np

emb1 = recognizer.get_normalized_embedding(
    image1,
    landmarks1
)

emb2 = recognizer.get_normalized_embedding(
    image2,
    landmarks2
)

similarity = compute_similarity(emb1, emb2)

print(f"Similarity: {similarity:.4f}")

Since the embeddings are L2-normalized, similarity can also be calculated directly:

similarity = np.dot(emb1, emb2)
📊 Similarity Threshold
Similarity	Decision	Use Case
> 0.70	🟢 Very high confidence	Security-critical applications
> 0.60	🟢 Same person	General verification
0.40–0.60	🟡 Uncertain	Manual review
< 0.40	🔴 Different person	Rejection

⚠️ The threshold should be calibrated for the selected recognition model and application.

🎯 Face Alignment

Recognition models require aligned faces for reliable embeddings.

UniFace performs alignment automatically when extracting an embedding:

embedding = recognizer.get_normalized_embedding(
    image,
    landmarks
)

Alignment can also be performed manually:

from uniface.face_utils import face_alignment

aligned_face, inverse_matrix = face_alignment(
    image,
    landmarks
)

The function returns:

🖼️ aligned_face — normalized 112 × 112 face image
🔄 inverse_matrix — transformation matrix used to map coordinates back to the original image
🗃️ Building a Face Database

Face embeddings can be stored and reused to create a simple face database.

import cv2
import numpy as np

from uniface.detection import RetinaFace
from uniface.recognition import ArcFace

detector = RetinaFace()
recognizer = ArcFace()

database = {}

for person_id, image_path in person_images.items():

    image = cv2.imread(image_path)

    faces = detector.detect(image)

    if not faces:
        continue

    face = faces[0]

    embedding = recognizer.get_normalized_embedding(
        image,
        face.landmarks
    )

    database[person_id] = embedding
💾 Save the Database
np.savez(
    "face_database.npz",
    **database
)
📂 Load the Database
data = np.load("face_database.npz")

database = {
    key: data[key]
    for key in data.files
}
🔎 Face Search

Search for the closest identity in the database.

def search_face(
    query_embedding,
    database,
    threshold=0.6
):
    """Return the closest identity above the threshold."""

    best_match = None
    best_similarity = -1.0

    for person_id, db_embedding in database.items():

        similarity = np.dot(
            query_embedding,
            db_embedding
        )

        if (
            similarity > best_similarity
            and similarity > threshold
        ):
            best_similarity = similarity
            best_match = person_id

    return best_match, best_similarity
Example
query_embedding = recognizer.get_normalized_embedding(
    query_image,
    landmarks
)

match, similarity = search_face(
    query_embedding,
    database
)

if match is not None:

    print(
        f"Match: {match} "
        f"(similarity: {similarity:.4f})"
    )

else:

    print("No matching face was found.")
📍 Facial Landmarks

Facial landmarks provide precise locations of important facial features such as the eyes, nose, mouth, eyebrows, and face contour.

Available Models
Model	Points	Type	Size
🔵 Landmark106	106	2D	14 MB
🟢 PIPNet WFLW	98	2D	47 MB
🟡 PIPNet 300W + CelebA	68	2D	46 MB
🔴 FaceMesh V1	468	3D	2.4 MB
🟣 FaceMesh V2	478	3D + Iris	4.6 MB
🔵 Landmark106
from uniface.detection import RetinaFace
from uniface.landmark import Landmark106

detector = RetinaFace()
landmarker = Landmark106()

faces = detector.detect(image)

if faces:

    landmarks = landmarker.get_landmarks(
        image,
        faces[0].bbox
    )

    print(
        f"Landmarks shape: {landmarks.shape}"
    )  # (106, 2)
Landmark Groups
Range	Group	Points
0–32	Face Contour	33
33–50	Eyebrows	18
51–62	Nose	12
63–86	Eyes	24
87–105	Mouth	19
Extract Specific Features
contour = landmarks[0:33]

left_eyebrow = landmarks[33:42]

right_eyebrow = landmarks[42:51]

nose = landmarks[51:63]

left_eye = landmarks[63:72]

right_eye = landmarks[76:84]

mouth = landmarks[87:106]
🟢 PIPNet — 98 / 68 Points

PIPNet is a high-accuracy facial landmark detector.

98-Point Model
from uniface.detection import RetinaFace
from uniface.landmark import PIPNet

detector = RetinaFace()
landmarker = PIPNet()

faces = detector.detect(image)

if faces:

    landmarks = landmarker.get_landmarks(
        image,
        faces[0].bbox
    )

    print(landmarks.shape)  # (98, 2)
68-Point Model
from uniface.constants import PIPNetWeights
from uniface.landmark import PIPNet

landmarker = PIPNet(
    model_name=PIPNetWeights.DW300_CELEBA_68
)

landmarks = landmarker.get_landmarks(
    image,
    face.bbox
)

print(landmarks.shape)  # (68, 2)
🕸️ FaceMesh — 468 / 478 Points

FaceMesh provides dense 3D facial landmarks.

from uniface import SCRFD, FaceMesh

detector = SCRFD()
mesher = FaceMesh()

faces = detector.detect(image)

results = mesher.predict(
    image,
    faces
)

results[0].landmarks.shape  # (468, 3)
results[0].points_2d.shape  # (468, 2)
results[0].score             # [0, 1]
👁️ 478-Point FaceMesh

The V2_478 model adds iris landmarks.

from uniface import SCRFD, FaceMesh
from uniface.constants import FaceMeshWeights
from uniface.landmark import (
    IRIS_LEFT,
    IRIS_RIGHT,
    NUM_MESH_LANDMARKS
)

mesher = FaceMesh(
    model_name=FaceMeshWeights.V2_478
)

result = mesher.predict(
    image,
    SCRFD().detect(image)
)[0]

print(result.landmarks.shape)
# (478, 3)
📌 5-Point Landmarks

All detection models except BlazeFace provide five facial landmarks:

👁️ Left eye
👁️ Right eye
👃 Nose
👄 Left mouth
👄 Right mouth
from uniface.detection import RetinaFace

detector = RetinaFace()

faces = detector.detect(image)

if faces:

    landmarks_5 = faces[0].landmarks

    print(landmarks_5.shape)
    # (5, 2)

    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    nose = landmarks_5[2]
    left_mouth = landmarks_5[3]
    right_mouth = landmarks_5[4]

⚠️ BlazeFace is different: it returns 6 MediaPipe keypoints instead of 5.

🎨 Landmark Visualization
import cv2

def draw_landmarks(
    image,
    landmarks,
    color=(0, 255, 0),
    radius=2
):

    for x, y in landmarks.astype(int):

        cv2.circle(
            image,
            (x, y),
            radius,
            color,
            -1
        )

    return image
Usage
landmarks = landmarker.get_landmarks(
    image,
    face.bbox
)

image_with_landmarks = draw_landmarks(
    image.copy(),
    landmarks
)

cv2.imwrite(
    "landmarks.jpg",
    image_with_landmarks
)
👁️ Eye Aspect Ratio — Blink Detection

Facial landmarks can also be used to detect eye closure and blinking.

import numpy as np

def eye_aspect_ratio(eye_landmarks):

    v1 = np.linalg.norm(
        eye_landmarks[1] -
        eye_landmarks[5]
    )

    v2 = np.linalg.norm(
        eye_landmarks[2] -
        eye_landmarks[4]
    )

    h = np.linalg.norm(
        eye_landmarks[0] -
        eye_landmarks[3]
    )

    ear = (v1 + v2) / (2.0 * h)

    return ear

Example:

left_eye = landmarks[63:72]

ear = eye_aspect_ratio(left_eye)

if ear < 0.2:
    print("Eye closed / Blink detected")
🧭 Head Pose Estimation

Facial landmarks can also be used to estimate the orientation of the head.

import cv2
import numpy as np

def estimate_head_pose(
    landmarks,
    image_shape
):

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    image_points = np.array([
        landmarks[51],
        landmarks[16],
        landmarks[63],
        landmarks[76],
        landmarks[87],
        landmarks[93]
    ], dtype=np.float64)

    h, w = image_shape[:2]

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs
    )

    return rotation_vector, translation_vector
🧰 Available Landmarkers
from uniface.constants import (
    FaceMeshWeights,
    PIPNetWeights
)

from uniface.landmark import (
    FaceMesh,
    Landmark106,
    PIPNet
)

# 🔵 106-point model
landmarker = Landmark106()

# 🟢 98-point PIPNet
landmarker = PIPNet()

# 🟡 68-point PIPNet
landmarker = PIPNet(
    model_name=PIPNetWeights.DW300_CELEBA_68
)

# 🔴 468-point FaceMesh
landmarker = FaceMesh()

# 🟣 478-point FaceMesh + Iris
landmarker = FaceMesh(
    model_name=FaceMeshWeights.V2_478
)
🚀 Complete Face Analysis Pipeline
📷 Input Image
      ↓
🔍 Face Detection
      ↓
📍 Facial Landmarks
      ↓
🎯 Face Alignment
      ↓
🧠 Face Recognition
      ↓
🔢 512-D Embedding
      ↓
🗃️ Face Database
      ↓
🔎 Face Search / Verification
      ↓
✅ Match / ❌ No Match

This provides a complete pipeline from input image → face detection → landmarks → alignment → embedding → database search → face verification.
