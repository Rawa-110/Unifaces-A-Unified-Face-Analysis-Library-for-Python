Face Recognition

UniFace provides a unified face-recognition interface for generating facial embeddings, comparing faces, verifying identities, and searching through a collection of known faces.

The recognition module supports several models with different balances between accuracy, speed, and model size.

Supported Recognition Models:

| Model          | Backbone            |      Size | Embedding |
| -------------- | ------------------- | --------: | --------: |
| **AdaFace**    | IR-18 / IR-101      | 92–249 MB |       512 |
| **ArcFace**    | MobileNet / ResNet  |  8–166 MB |       512 |
| **EdgeFace**   | EdgeNeXt / LoRA     |   5–70 MB |       512 |
| **MobileFace** | MobileNet V2 / V3   |   1–10 MB |       512 |
| **SphereFace** | Sphere20 / Sphere36 |  50–92 MB |       512 |


All recognition models generate a 512-dimensional normalized embedding, which can then be used for face comparison or database search.

AdaFace

AdaFace is a face-recognition model that adapts its margin according to the quality of the input image. This makes it particularly useful when face images have different levels of quality.

Basic Usage

from uniface.detection import RetinaFace
from uniface.recognition import AdaFace

detector = RetinaFace()
recognizer = AdaFace()

# Detect faces
faces = detector.detect(image)

# Generate an embedding
if faces:
    embedding = recognizer.get_normalized_embedding(
        image,
        faces[0].landmarks
    )

    print(f"Embedding shape: {embedding.shape}")


:

(512,)
Model Options

AdaFace provides two main variants:

from uniface.recognition import AdaFace
from uniface.constants import AdaFaceWeights

# Smaller and faster
recognizer = AdaFace(
    model_name=AdaFaceWeights.IR_18
)

# Larger model with higher accuracy
recognizer = AdaFace(
    model_name=AdaFaceWeights.IR_101
)

For CPU-only execution:
recognizer = AdaFace(
    providers=["CPUExecutionProvider"]
)

| Variant    | Training Dataset |   Size |  IJB-B |  IJB-C |
| ---------- | ---------------- | -----: | -----: | -----: |
| **IR_18**  | WebFace4M        |  92 MB | 93.03% | 94.99% |
| **IR_101** | WebFace12M       | 249 MB |      — | 97.66% |

The reported IJB-B and IJB-C values use TAR at FAR = 0.01%.

ArcFace

ArcFace is a widely used face-recognition approach based on an additive angular margin loss. It provides strong recognition accuracy while supporting both lightweight and larger backbones.

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

    print(f"Embedding shape: {embedding.shape}")
Model Variants
from uniface.recognition import ArcFace
from uniface.constants import ArcFaceWeights

# Lightweight option
recognizer = ArcFace(
    model_name=ArcFaceWeights.MNET
)

# Higher-accuracy option
recognizer = ArcFace(
    model_name=ArcFaceWeights.RESNET
)

CPU execution can be forced with:

recognizer = ArcFace(
    providers=["CPUExecutionProvider"]
)

| Variant    | Backbone  |   Size |    LFW | CFP-FP | AgeDB-30 |  IJB-C |
| ---------- | --------- | -----: | -----: | -----: | -------: | -----: |
| **MNET**   | MobileNet |   8 MB | 99.70% | 98.00% |   96.58% | 95.02% |
| **RESNET** | ResNet50  | 166 MB | 99.83% | 99.33% |   98.23% | 97.25% |

ArcFace models are trained using the WebFace600K dataset.

EdgeFace

-based architecture and includes compact variants suitable for edge and mobile applications.

Basic Usage
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

    print(f"Embedding shape: {embedding.shape}")
Model Variants
from uniface.recognition import EdgeFace
from uniface.constants import EdgeFaceWeights

# Very small model
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.XXS
)

# Compact LoRA variant
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.XS_GAMMA_06
)

# Small LoRA variant
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.S_GAMMA_05
)

# Full-size model
recognizer = EdgeFace(
    model_name=EdgeFaceWeights.BASE
)

Performance
Variant	Params	MFLOPs	Size	LFW	CALFW	CPLFW	CFP-FP	AgeDB-30
XXS	1.24M	94	~5 MB	99.57%	94.83%	90.27%	93.63%	94.92%
XS_GAMMA_06	1.77M	154	~7 MB	99.73%	95.28%	91.58%	94.71%	96.08%
S_GAMMA_05	3.65M	306	~14 MB	99.78%	95.55%	92.48%	95.74%	97.03%
BASE	18.2M	1399	~70 MB	99.83%	96.07%	93.75%	97.01%	97.60%

Reference: EdgeFace: Efficient Face Recognition Model for Edge Devices.

MobileFace

MobileFace focuses on low-resource face recognition using MobileNet-based architectures. It is a suitable choice when inference speed and small model size are important.

Basic Usage

from uniface.recognition import MobileFace

recognizer = MobileFace()

embedding = recognizer.get_normalized_embedding(
    image,
    landmarks
)

from uniface.recognition import MobileFace
from uniface.constants import MobileFaceWeights

# Extremely lightweight
recognizer = MobileFace(
    model_name=MobileFaceWeights.MNET_025
)

# Balanced configuration
recognizer = MobileFace(
    model_name=MobileFaceWeights.MNET_V2
)

# Higher-capacity model
recognizer = MobileFace(
    model_name=MobileFaceWeights.MNET_V3_LARGE
)
Performance
| Variant           | Params |  Size |    LFW |  CALFW |  CPLFW | AgeDB-30 |
| ----------------- | -----: | ----: | -----: | -----: | -----: | -------: |
| **MNET_025**      |  0.36M |  1 MB | 98.76% | 92.02% | 82.37% |   90.02% |
| **MNET_V2**       |  2.29M |  4 MB | 99.55% | 94.87% | 86.89% |   95.16% |
| **MNET_V3_SMALL** |  1.25M |  3 MB | 99.30% | 93.77% | 85.29% |   92.79% |
| **MNET_V3_LARGE** |  3.52M | 10 MB | 99.53% | 94.56% | 86.79% |   95.13% |

from uniface.recognition import SphereFace
from uniface.constants import SphereFaceWeights

recognizer = SphereFace(
    model_name=SphereFaceWeights.SPHERE20
)

embedding = recognizer.get_normalized_embedding(
    image,
    landmarks
)
Model Variants
| Variant      | Params |  Size |    LFW |  CALFW |  CPLFW | AgeDB-30 |
| ------------ | -----: | ----: | -----: | -----: | -----: | -------: |
| **SPHERE20** |  24.5M | 50 MB | 99.67% | 95.61% | 88.75% |   96.58% |
| **SPHERE36** |  34.6M | 92 MB | 99.72% | 95.64% | 89.92% |   96.83% |

Once two faces have been converted into embeddings, their similarity can be calculated.

UniFace provides a utility function for this purpose:

from uniface.face_utils import compute_similarity

similarity = compute_similarity(
    embedding1,
    embedding2
)

print(f"Similarity: {similarity:.4f}")

Because the embeddings are normalized, cosine similarity can also be calculated directly using the dot product:

import numpy as np

similarity = np.dot(
    embedding1,
    embedding2
)

print(f"Similarity: {similarity:.4f}")
Similarity Thresholds

The similarity score can be used to determine whether two embeddings are likely to belong to the same identity.

Similarity	Interpretation	Recommended Action
> 0.70	Very strong similarity	High-confidence verification
> 0.60	Strong similarity	General verification
0.40–0.60	Ambiguous	Manual review
< 0.40	Low similarity	Likely different identities

Important: These values are general guidelines. The appropriate threshold should be calibrated for the selected model, dataset, image quality, and application.

Face Alignment

Recognition models work best when faces are properly aligned. UniFace can perform this step automatically when generating embeddings.

embedding = recognizer.get_normalized_embedding(
    image,
    landmarks
)

from uniface.face_utils import face_alignment

aligned_face, inverse_matrix = face_alignment(
    image,
    landmarks
)

The function returns:

aligned_face — a normalized 112 × 112 face image.
inverse_matrix — the transformation matrix used to map coordinates back to the original image.
Building a Face Database

Face embeddings can be stored as numerical vectors and reused later for identity searches.

The following example detects a face from each image and stores its embedding:

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



Saving the Face Database

The generated embeddings can be saved with NumPy:

np.savez(
    "face_database.npz",
    **database
)

This allows the embeddings to be reused without running the recognition model again.

Loading the Database

The stored database can later be restored:

data = np.load("face_database.npz")

database = {
    key: data[key]
    for key in data.files
}
Face Search

After creating the database, a new face can be compared against all stored embeddings.

The following function returns the closest identity whose similarity exceeds the selected threshold:

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

Generate an embedding for the query image:

query_embedding = recognizer.get_normalized_embedding(
    query_image,
    landmarks
)

Then search the database:

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
Switching Between Recognition Models

One advantage of UniFace is that the recognition workflow remains almost identical across different backends.

from uniface.recognition import (
    ArcFace,
    AdaFace,
    EdgeFace,
    MobileFace,
    SphereFace,
)

You can select the model that best fits your application:

recognizer = ArcFace()

or:

recognizer = AdaFace()

or:

recognizer = EdgeFace()

or:

recognizer = MobileFace()

or:

recognizer = SphereFace()

All of these models follow the same general workflow:

Image → Face Detection → Landmarks → Alignment → Embedding → Similarity → Verification/Search

This makes it possible to change the recognition backend without redesigning the entire face-recognition pipeline.

Recommended Workflow

For a complete face-recognition system, the typical process is:

Input Image
     ↓
Face Detection
     ↓
Facial Landmarks
     ↓
Face Alignment
     ↓
Recognition Model
     ↓
512-D Embedding
     ↓
Cosine Similarity
     ↓
Threshold
     ↓
Match / No Match

Example
import cv2
import numpy as np

from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.face_utils import compute_similarity

detector = RetinaFace()
recognizer = ArcFace()

image1 = cv2.imread("person1.jpg")
image2 = cv2.imread("person2.jpg")

faces1 = detector.detect(image1)
faces2 = detector.detect(image2)

if faces1 and faces2:

    emb1 = recognizer.get_normalized_embedding(
        image1,
        faces1[0].landmarks
    )

    emb2 = recognizer.get_normalized_embedding(
        image2,
        faces2[0].landmarks
    )

    similarity = compute_similarity(
        emb1,
        emb2
    )

    print(f"Similarity: {similarity:.4f}")

    if similarity >= 0.6:
        print("Same person")
    else:
        print("Different people")

Note: A similarity score should not be treated as absolute proof of identity. Thresholds need to be validated on data representative of the intended application, especially for security-sensitive use cases.

