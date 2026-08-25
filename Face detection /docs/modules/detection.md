# Detection

Face detection is the first step in any face-analysis pipeline. UniFace provides six detection models:

- RetinaFace
- SCRFD
- CenterFace
- YOLOv5-Face
- YOLOv8-Face
- BlazeFace

---

## Available Models

| Model | Backbone | Size | Easy | Medium | Hard | Landmarks |
|---|---|---:|---:|---:|---:|:---:|
| RetinaFace | MobileNet V2 | 3.5 MB | 91.7% | 91.0% | 86.6% | ✅ |
| SCRFD | SCRFD-10G | 17 MB | 95.2% | 93.9% | 83.1% | ✅ |
| CenterFace | MobileNet V2 | 7.0 MB | 92.2% | 91.1% | 78.2% | ✅ |
| YOLOv5-Face | YOLOv5s | 28 MB | 94.3% | 92.6% | 83.2% | ✅ |
| YOLOv8-Face | YOLOv8n | 12 MB | 94.6% | 92.3% | 79.6% | ✅ |
| BlazeFace | BlazeFace | 0.5 MB | — | — | — | 6-point |

> **Dataset:** All models except BlazeFace are trained on the WIDER FACE dataset and benchmarked against it. BlazeFace comes from Google MediaPipe and is not benchmarked on WIDER FACE.

### Landmark Compatibility

All detectors except BlazeFace return the standard 5-point alignment template:

```text
[left_eye, right_eye, nose, left_mouth, right_mouth]

.

BlazeFace returns six MediaPipe keypoints instead:

[right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear]

Therefore, BlazeFace does not support the standard alignment pipeline.

detector.supports_alignment
# False for BlazeFace
# True for other built-in detectors

detector.supports_landmarks
# True for all built-in detectors
RetinaFace

RetinaFace is a single-shot face detector based on a multi-scale feature pyramid.

Basic Usage
from uniface.detection import RetinaFace

detector = RetinaFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"BBox: {face.bbox}")
    print(f"Landmarks: {face.landmarks.shape}")  # (5, 2)
Model Variants
from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights

# Lightweight
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_025
)

# Balanced (default)
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2
)

# High accuracy
detector = RetinaFace(
    model_name=RetinaFaceWeights.RESNET34
)
Variant	Params	Size	Easy	Medium	Hard
MNET_025	0.4M	1.7 MB	88.5%	87.0%	80.6%
MNET_050	1.0M	2.6 MB	89.4%	88.0%	82.4%
MNET_V1	3.5M	3.8 MB	90.6%	89.1%	84.1%
MNET_V2	3.2M	3.5 MB	91.7%	91.0%	86.6%
RESNET18	11.7M	27 MB	92.5%	91.0%	86.6%
RESNET34	24.8M	56 MB	94.2%	93.1%	88.9%
RESNET50	27.4M	104 MB	94.7%*	93.7%*	88.8%*

Note: RESNET50 weights come from HivisionIDPhotos. Its scores are measured rather than quoted from the original paper.

Configuration
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    input_size=(640, 640),
    dynamic_size=False,
    providers=None,
)
SCRFD

SCRFD provides an excellent balance between detection accuracy and inference speed.

Basic Usage
from uniface.detection import SCRFD

detector = SCRFD()
faces = detector.detect(image)
Model Variants
from uniface.detection import SCRFD
from uniface.constants import SCRFDWeights

# Lightweight / real-time
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_500M_KPS
)

# High accuracy (default)
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS
)
Variant	Params	Size	Easy	Medium	Hard
SCRFD_500M_KPS	0.6M	2.5 MB	90.6%	88.1%	68.5%
SCRFD_10G_KPS	4.2M	17 MB	95.2%	93.9%	83.1%
Configuration
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    input_size=(640, 640),
    providers=None,
)
CenterFace

CenterFace is an anchor-free detector that represents faces as center points.

It uses MobileNetV2 + FPN and provides joint 5-point landmark prediction.

Paper: CenterFace: Joint Face Detection and Alignment Using Face as Point

Basic Usage
from uniface.detection import CenterFace

detector = CenterFace()
faces = detector.detect(image)
Variant	Size	Easy	Medium	Hard
DEFAULT	7.0 MB	92.2%	91.1%	78.2%

Benchmark: Scores are WIDER FACE validation results using single inference on the original image.

Limitations
Landmark precision is lower than SCRFD and RetinaFace.
Landmark accuracy decreases as face rotation increases.
Best suited for relatively upright faces.
For alignment-critical recognition, SCRFD or RetinaFace is recommended.
Configuration
from uniface.constants import CenterFaceWeights

detector = CenterFace(
    model_name=CenterFaceWeights.DEFAULT,
    confidence_threshold=0.35,
    nms_threshold=0.3,
    input_size=(640, 640),
    providers=None,
)
YOLOv5-Face

YOLOv5-Face is a YOLO-based detector optimized for face detection.

Basic Usage
from uniface.detection import YOLOv5Face

detector = YOLOv5Face()
faces = detector.detect(image)
Model Variants
from uniface.detection import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights

# Lightweight
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5N
)

# Balanced (default)
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S
)

# High accuracy
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5M
)
Variant	Size	Easy	Medium	Hard
YOLOV5N	11 MB	93.6%	91.5%	80.5%
YOLOV5S	28 MB	94.3%	92.6%	83.2%
YOLOV5M	82 MB	95.3%	93.8%	85.3%

Note: YOLOv5-Face uses a fixed input size of 640×640.

Configuration
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    confidence_threshold=0.6,
    nms_threshold=0.5,
    nms_mode="numpy",
    providers=None,
)
YOLOv8-Face

YOLOv8-Face is an anchor-free detector using Distribution Focal Loss (DFL) for accurate bounding-box regression.

Basic Usage
from uniface.detection import YOLOv8Face

detector = YOLOv8Face()
faces = detector.detect(image)
Model Variants
from uniface.detection import YOLOv8Face
from uniface.constants import YOLOv8FaceWeights

# Lightweight
detector = YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8_LITE_S
)

# Recommended
detector = YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8N
)
Variant	Size	Easy	Medium	Hard
YOLOV8_LITE_S	7.4 MB	93.4%	91.2%	78.6%
YOLOV8N	12 MB	94.6%	92.3%	79.6%

Note: YOLOv8-Face uses a fixed input size of 640×640.

Configuration
detector = YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8N,
    confidence_threshold=0.5,
    nms_threshold=0.45,
    nms_mode="numpy",
    providers=None,
)
BlazeFace

BlazeFace is Google's short-range SSD face detector used in the MediaPipe pipeline.

At only 0.5 MB, it is the smallest detector in UniFace.

It is designed primarily for faces within approximately 2 meters and is optimized for lightweight and real-time applications.

Basic Usage
from uniface.detection import BlazeFace

detector = BlazeFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"Keypoints: {face.landmarks.shape}")  # (6, 2)
Keypoint Layout
[right_eye,
 left_eye,
 nose_tip,
 mouth_center,
 right_ear,
 left_ear]

Unlike other UniFace detectors, BlazeFace does not return the standard 5-point alignment template.

detector.supports_alignment
# False
Configuration
detector = BlazeFace(
    confidence_threshold=0.5,
    nms_threshold=0.3,
    providers=None,
)
Available Detectors
from uniface.detection import (
    BlazeFace,
    CenterFace,
    RetinaFace,
    SCRFD,
    YOLOv5Face,
    YOLOv8Face,
)

detector = RetinaFace()

# Or:
detector = SCRFD()
detector = CenterFace()
detector = YOLOv5Face()
detector = YOLOv8Face()
detector = BlazeFace()
Output Format

All detectors return:

list[Face]

Each Face object contains:

for face in faces:

    # Bounding box
    bbox = face.bbox

    # Detection confidence
    confidence = face.confidence

    # Facial landmarks
    landmarks = face.landmarks

For standard detectors:

landmarks.shape == (5, 2)

The five points are:

[left_eye,
 right_eye,
 nose,
 left_mouth,
 right_mouth]

BlazeFace returns six keypoints:

landmarks.shape == (6, 2)
Visualization

UniFace provides a built-in visualization helper:

from uniface.draw import draw_detections

draw_detections(
    image=image,
    faces=faces,
    vis_threshold=0.6,
)

cv2.imwrite("result.jpg", image)
Performance Comparison

Run the detection benchmark on your own hardware:

python tools/detect.py --source image.jpg

Performance depends on:

Hardware
Model variant
Input resolution
Confidence threshold
Number of faces
Execution provider
See Also
Recognition — Extract embeddings from detected faces
Landmarks — Generate dense facial landmarks
Image Pipeline — Complete detection workflow
Threshold Calibration — Tune detection parameters
CLI Tools — Command-line utilities

**بهذا الشكل** يصبح القسم نظيفًا جدًا داخل GitHub، ولا يحتوي على عناصر `svg` الغريبة أو الروابط المتكررة الناتجة من نسخ صفحة التوثيق. كما أن كل Model أصبح له نفس الهيكل: **Description → Basic Usage → Variants → Configuration**.
عيد الصياغه غير في كلام بالانجليزي

أكيد. هذه إعادة صياغة بالإنجليزية بالكامل مع تغيير الأسلوب والكلمات، مع الحفاظ على نفس المعلومات التقنية. جعلتها أكثر احترافية ومناسبة لصفحة Documentation أو GitHub README.

# Face Detection

Face detection is the foundation of most facial-analysis systems. It identifies facial regions in an image or video frame and provides the coordinates and key facial points required by subsequent processing stages.

UniFace includes six built-in detection backends, ranging from extremely lightweight solutions for edge devices to higher-accuracy models for challenging scenes.

## Supported Detection Models

| Model | Architecture | Model Size | Easy | Medium | Hard | Keypoints |
|---|---|---:|---:|---:|---:|:---:|
| RetinaFace | MobileNet V2 | 3.5 MB | 91.7% | 91.0% | 86.6% | 5-point |
| SCRFD | SCRFD-10G | 17 MB | 95.2% | 93.9% | 83.1% | 5-point |
| CenterFace | MobileNet V2 | 7.0 MB | 92.2% | 91.1% | 78.2% | 5-point |
| YOLOv5-Face | YOLOv5s | 28 MB | 94.3% | 92.6% | 83.2% | 5-point |
| YOLOv8-Face | YOLOv8n | 12 MB | 94.6% | 92.3% | 79.6% | 5-point |
| BlazeFace | MediaPipe BlazeFace | 0.5 MB | — | — | — | 6-point |

> **Benchmark:** Except for BlazeFace, the listed detectors are evaluated on the WIDER FACE benchmark. BlazeFace originates from the Google MediaPipe ecosystem and follows a different evaluation setup.

---

## Landmark Compatibility

Most UniFace detectors generate five facial reference points that can be used for alignment and recognition:

```text
Left Eye
Right Eye
Nose
Left Mouth Corner
Right Mouth Corner

These points provide a common alignment format for recognition, quality assessment, and facial parsing.

BlazeFace follows the MediaPipe keypoint convention and produces six points:

Right Eye
Left Eye
Nose Tip
Mouth Center
Right Ear
Left Ear

Because this layout differs from the standard alignment template, BlazeFace cannot be directly used by the alignment-dependent recognition pipeline.

detector.supports_alignment
# False for BlazeFace
# True for the other built-in detectors

detector.supports_landmarks
# True for all built-in detectors
RetinaFace

RetinaFace is a single-stage detector designed to locate faces at different scales. It combines a feature pyramid with dedicated face and landmark prediction heads, making it suitable for both general-purpose detection and downstream facial analysis.

Quick Example
from uniface.detection import RetinaFace

detector = RetinaFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"Bounding Box: {face.bbox}")
    print(f"Landmarks: {face.landmarks.shape}")

The standard RetinaFace output contains five facial landmarks:

(5, 2)
Available Variants

RetinaFace provides several model sizes so you can choose between speed, memory usage, and accuracy.

from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights

# Smallest configuration
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_025
)

# Recommended general-purpose configuration
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2
)

# Higher-accuracy configuration
detector = RetinaFace(
    model_name=RetinaFaceWeights.RESNET34
)
Variant	Parameters	Size	Easy	Medium	Hard
MNET_025	0.4M	1.7 MB	88.5%	87.0%	80.6%
MNET_050	1.0M	2.6 MB	89.4%	88.0%	82.4%
MNET_V1	3.5M	3.8 MB	90.6%	89.1%	84.1%
MNET_V2	3.2M	3.5 MB	91.7%	91.0%	86.6%
RESNET18	11.7M	27 MB	92.5%	91.0%	86.6%
RESNET34	24.8M	56 MB	94.2%	93.1%	88.9%
RESNET50	27.4M	104 MB	94.7%*	93.7%*	88.8%*

RESNET50 note: The RESNET50 weights are sourced from HivisionIDPhotos. The reported values are measured results rather than numbers directly taken from the original RetinaFace publication.

Configuration
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    input_size=(640, 640),
    dynamic_size=False,
    providers=None,
)
Configuration Parameters
Parameter	Description
model_name	Selects the RetinaFace model variant
confidence_threshold	Minimum confidence required for a detection
nms_threshold	IoU threshold used during non-maximum suppression
input_size	Resolution used during inference
dynamic_size	Enables dynamic input dimensions
providers	ONNX Runtime execution providers
SCRFD

SCRFD is designed to provide a strong balance between inference speed and detection accuracy. It is particularly useful when real-time performance is important without sacrificing too much detection quality.

Quick Example
from uniface.detection import SCRFD

detector = SCRFD()
faces = detector.detect(image)
Model Variants
from uniface.detection import SCRFD
from uniface.constants import SCRFDWeights

# Lightweight model
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_500M_KPS
)

# Higher-accuracy model
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS
)
Variant	Parameters	Size	Easy	Medium	Hard
SCRFD_500M_KPS	0.6M	2.5 MB	90.6%	88.1%	68.5%
SCRFD_10G_KPS	4.2M	17 MB	95.2%	93.9%	83.1%
Configuration
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    input_size=(640, 640),
    providers=None,
)
CenterFace

CenterFace uses an anchor-free detection strategy in which faces are represented by their center locations. The model is based on MobileNetV2 and FPN and also predicts five facial landmarks.

It is particularly attractive for CPU-based applications where low computational overhead is important.

Quick Example
from uniface.detection import CenterFace

detector = CenterFace()
faces = detector.detect(image)
Variant	Size	Easy	Medium	Hard
DEFAULT	7.0 MB	92.2%	91.1%	78.2%
Considerations

CenterFace is lightweight and fast, but its landmark precision is generally lower than RetinaFace or SCRFD.

For applications where facial alignment has a major impact on recognition quality, consider using RetinaFace or SCRFD instead.

Performance may also decrease when faces are strongly rotated. CenterFace is best suited to relatively frontal or upright faces.

Configuration
from uniface.constants import CenterFaceWeights

detector = CenterFace(
    model_name=CenterFaceWeights.DEFAULT,
    confidence_threshold=0.35,
    nms_threshold=0.3,
    input_size=(640, 640),
    providers=None,
)
YOLOv5-Face

YOLOv5-Face adapts the YOLO object-detection architecture specifically for facial detection.

It offers several model sizes, allowing the detector to be configured for different hardware and accuracy requirements.

Quick Example
from uniface.detection import YOLOv5Face

detector = YOLOv5Face()
faces = detector.detect(image)
Model Variants
from uniface.detection import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights

# Lightweight
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5N
)

# Recommended balance
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S
)

# Larger model
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5M
)
Variant	Size	Easy	Medium	Hard
YOLOV5N	11 MB	93.6%	91.5%	80.5%
YOLOV5S	28 MB	94.3%	92.6%	83.2%
YOLOV5M	82 MB	95.3%	93.8%	85.3%

Input resolution: YOLOv5-Face uses a fixed inference resolution of 640 × 640.

Configuration
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    confidence_threshold=0.6,
    nms_threshold=0.5,
    nms_mode="numpy",
    providers=None,
)
YOLOv8-Face

YOLOv8-Face uses an anchor-free detection design together with Distribution Focal Loss (DFL) for bounding-box regression.

It provides a compact option for applications that require modern YOLO-style inference.

Quick Example
from uniface.detection import YOLOv8Face

detector = YOLOv8Face()
faces = detector.detect(image)
Model Variants
from uniface.detection import YOLOv8Face
from uniface.constants import YOLOv8FaceWeights

# Lightweight configuration
detector = YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8_LITE_S
)

# Recommended configuration
detector = YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8N
)
Variant	Size	Easy	Medium	Hard
YOLOV8_LITE_S	7.4 MB	93.4%	91.2%	78.6%
YOLOV8N	12 MB	94.6%	92.3%	79.6%

Input resolution: YOLOv8-Face uses a fixed inference resolution of 640 × 640.

Configuration
detector = YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8N,
    confidence_threshold=0.5,
    nms_threshold=0.45,
    nms_mode="numpy",
    providers=None,
)
BlazeFace

BlazeFace is a compact short-range face detector originally developed for the MediaPipe ecosystem.

With a model size of approximately 0.5 MB, it is the smallest detector available in UniFace.

BlazeFace is optimized for relatively close faces and lightweight real-time applications. It is not intended to replace larger detectors when detecting small or distant faces is the main requirement.

Quick Example
from uniface.detection import BlazeFace

detector = BlazeFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"Keypoints: {face.landmarks.shape}")
Keypoint Layout

BlazeFace produces six MediaPipe-style keypoints:

[right_eye,
 left_eye,
 nose_tip,
 mouth_center,
 right_ear,
 left_ear]

This differs from the standard five-point alignment representation used by the other detectors.

detector.supports_alignment
# False
Configuration
detector = BlazeFace(
    confidence_threshold=0.5,
    nms_threshold=0.3,
    providers=None,
)
Choosing a Detector

The best detector depends on your application:

Requirement	Recommended Model
General-purpose detection	RetinaFace / SCRFD
High detection accuracy	SCRFD 10G
Lightweight CPU inference	CenterFace
YOLO-based pipeline	YOLOv5-Face / YOLOv8-Face
Very small model	BlazeFace
Face recognition pipeline	RetinaFace / SCRFD
Mobile / edge deployment	RetinaFace Mobile / SCRFD 500M
Available Detectors

All built-in detectors can be imported from uniface.detection:

from uniface.detection import (
    RetinaFace,
    SCRFD,
    CenterFace,
    YOLOv5Face,
    YOLOv8Face,
    BlazeFace,
)

detector = RetinaFace()

You can switch between detectors without changing the rest of your application pipeline.

Output Format

Every detector returns a list of Face objects:

faces = detector.detect(image)

for face in faces:
    bbox = face.bbox
    confidence = face.confidence
    landmarks = face.landmarks
Bounding Box

The bounding box follows the format:

[x1, y1, x2, y2]
Confidence

The detection confidence is represented as a value between 0 and 1.

Landmarks

Standard detectors return five points:

[left_eye,
 right_eye,
 nose,
 left_mouth,
 right_mouth]

BlazeFace returns six points using the MediaPipe layout.

Visualization

Detection results can be rendered using the built-in drawing utilities:

from uniface.draw import draw_detections

draw_detections(
    image=image,
    faces=faces,
    vis_threshold=0.6,
)

cv2.imwrite("result.jpg", image)
Performance Testing

To evaluate detection performance on your own system, run:

python tools/detect.py --source image.jpg

Actual performance depends on several factors, including:

Hardware
CPU/GPU backend
Selected model
Input resolution
Number of faces
Confidence threshold
Execution provider
Related Modules
Recognition — Generate and compare face embeddings
Landmarks — Extract dense facial landmarks
Image Pipeline — Build a complete face-analysis workflow
Threshold Calibration — Adjust detection thresholds
CLI Tools — Command-line utilities for UniFace
