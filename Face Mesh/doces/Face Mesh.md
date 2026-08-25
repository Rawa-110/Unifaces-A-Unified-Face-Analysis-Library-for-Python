🔷 Face Mesh — 468 or 478 3D Points

Face Mesh provides dense facial landmark detection with 468 or 478 three-dimensional points. It is useful for detailed face analysis, eye/iris tracking, facial alignment, and head-pose estimation

✨ Model Variants:
| Model    | Points | Type                  |   Size |
| -------- | -----: | --------------------- | -----: |
| `V1_468` |    468 | 3D landmarks          | 2.4 MB |
| `V2_478` |    478 | 3D landmarks + irises | 4.6 MB |

🚀 Basic Usage

from uniface import SCRFD, FaceMesh

# Initialize detector and Face Mesh
detector = SCRFD()
mesher = FaceMesh()

# Detect faces
faces = detector.detect(image)

# Extract Face Mesh landmarks
results = mesher.predict(image, faces)

# Get the first face
if results:
    result = results[0]

    print("Landmarks:", result.landmarks.shape)  # (468, 3)
    print("2D Points:", result.points_2d.shape)  # (468, 2)
    print("Face Score:", result.score)

    👁️ 478-Point Model with Iris Detection

The V2_478 model adds 10 iris landmarks to the standard 468-point mesh
from uniface import SCRFD, FaceMesh
from uniface.constants import FaceMeshWeights

detector = SCRFD()
mesher = FaceMesh(
    model_name=FaceMeshWeights.V2_478
)

faces = detector.detect(image)
results = mesher.predict(image, faces)

if results:
    result = results[0]

    print(result.landmarks.shape)  # (478, 3)


<img width="2687" height="3356" alt="image" src="https://github.com/user-attachments/assets/fb046f4e-b9d2-491f-8c06-2c3e9453cf45" />






    

    🔍 Access the Iris Points
    from uniface.constants import (
    IRIS_LEFT,
    IRIS_RIGHT,
    NUM_MESH_LANDMARKS
)

# Original 468 mesh points
mesh_points = result.landmarks[:NUM_MESH_LANDMARKS]

# Left iris
left_iris = result.landmarks[IRIS_LEFT]

# Right iris
right_iris = result.landmarks[IRIS_RIGHT]

print(left_iris.shape)   # (5, 3)
print(right_iris.shape)  # (5, 3)

Each iris contains 5 points:
        Top
         ●
         |
Left ● — ● — ● Right
       Center
         |
         ●
       Bottom

       The first point is the iris center, while the remaining four points represent the surrounding iris.

📐 3D Output

The landmarks are returned as:
result.landmarks

with the shape:

(468, 3)

or:

(478, 3)

The three values represent:

[x, y, z]
x → horizontal position
y → vertical position
z → relative depth

If you only need 2D coordinates:
points_2d = result.points_2d

Output:

(468, 2)

or:

(478, 2)
🎨 Visualization
from uniface.draw import draw_mesh

# Partial mesh — recommended for real-time applications
draw_mesh(
    image,
    result.landmarks,
    mode="partial"
)

# Full mesh
draw_mesh(
    image,
    result.landmarks,
    mode="full"
)

# Points only
draw_mesh(
    image,
    result.landmarks,
    mode="points"
)
⚡ Recommended Modes
Mode	Description	Performance
partial	Face contours + points	⭐⭐⭐
points	Points only	⭐⭐⭐
full	Complete facial mesh	⭐

⚠️ full draws 2556 connections per face, so partial or points is better for real-time video.

🎯 Using Face Bounding Boxes Directly

You can use Face Mesh without running a detector first:

results = mesher.predict(
    image,
    bboxes=[
        [x1, y1, x2, y2]
    ]
)

However, using detected Face objects is preferable for tilted faces because the crop can be automatically aligned according to the eye positions.

🧩 Drop-in Landmark Usage

Face Mesh can also be used similarly to other landmark models:

landmarks = mesher.get_landmarks(
    image,
    face.bbox
)

print(landmarks.shape)

For V2_478:

(478, 2)
📦 Complete Example
import cv2

from uniface import SCRFD, FaceMesh
from uniface.constants import FaceMeshWeights
from uniface.draw import draw_mesh

# Load image
image = cv2.imread("photo.jpg")

# Initialize models
detector = SCRFD()
mesher = FaceMesh(
    model_name=FaceMeshWeights.V2_478
)

# Detect faces
faces = detector.detect(image)

# Generate Face Mesh
results = mesher.predict(
    image,
    faces
)

# Draw mesh
for result in results:
    draw_mesh(
        image,
        result.landmarks,
        mode="partial"
    )

# Save result
cv2.imwrite(
    "face_mesh_result.jpg",
    image
)
📝 Summary
SCRFD
  ↓
Face Detection
  ↓
Face Mesh
  ↓
468 / 478 3D Landmarks
  ↓
Face Analysis

Use V1_468 when you need the standard dense face mesh.

Use V2_478 when you also need iris landmarks for applications such as eye tracking or detailed eye analysis




