All models except BlazeFace are trained on the WIDER FACE dataset and benchmarked against it.

BlazeFace comes from Google MediaPipe and is not benchmarked on WIDER FACE.

Landmarks Compatibility

Most detectors return a 5-point facial alignment template:

left eye
right eye
nose
left mouth corner
right mouth corner

These landmarks are used by recognition, face quality, and XSeg parsing.

BlazeFace is different: it returns 6 MediaPipe keypoints, including the mouth center and ear points. Therefore:

detector.supports_alignment
# False for BlazeFace
# True for other built-in detectors

detector.supports_landmarks
# True for all built-in detectors
RetinaFace

RetinaFace is a single-shot face detector using a multi-scale feature pyramid.

Basic Usage
from uniface.detection import RetinaFace

detector = RetinaFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"BBox: {face.bbox}")
    print(f"Landmarks: {face.landmarks.shape}")
Model Variants
from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights

# Lightweight
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_025
)

# Balanced
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

SCRFD provides a strong balance between detection accuracy and inference speed.

Basic Usage
from uniface.detection import SCRFD

detector = SCRFD()
faces = detector.detect(image)
Model Variants
from uniface.detection import SCRFD
from uniface.constants import SCRFDWeights

# Lightweight
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_500M_KPS
)

# High accuracy
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

It uses MobileNetV2 + FPN and predicts 5 facial landmarks alongside detection.

Basic Usage
from uniface.detection import CenterFace

detector = CenterFace()
faces = detector.detect(image)
Configuration
from uniface.constants import CenterFaceWeights

detector = CenterFace(
    model_name=CenterFaceWeights.DEFAULT,
    confidence_threshold=0.35,
    nms_threshold=0.3,
    input_size=(640, 640),
    providers=None,
)

Recommended for: lightweight CPU inference and relatively upright faces.

YOLOv5-Face

YOLOv5-Face is a YOLO-based face detector optimized for face detection.

Basic Usage
from uniface.detection import YOLOv5Face

detector = YOLOv5Face()
faces = detector.detect(image)
Model Variants
from uniface.detection import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights

detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S
)
Variant	Size	Easy	Medium	Hard
YOLOV5N	11 MB	93.6%	91.5%	80.5%
YOLOV5S	28 MB	94.3%	92.6%	83.2%
YOLOV5M	82 MB	95.3%	93.8%	85.3%

YOLOv5-Face uses a fixed input size of 640×640.

YOLOv8-Face

YOLOv8-Face uses anchor-free detection with DFL for accurate bounding-box regression.

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

YOLOv8-Face uses a fixed input size of 640×640.

BlazeFace

BlazeFace is Google's MediaPipe short-range face detector.

It is extremely lightweight at approximately 0.5 MB, making it suitable for mobile and real-time applications.

Basic Usage
from uniface.detection import BlazeFace

detector = BlazeFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"Keypoints: {face.landmarks.shape}")

BlazeFace returns 6 keypoints instead of the standard 5-point alignment template.

[right_eye, left_eye, nose_tip,
 mouth_center, right_ear, left_ear]

Because these points are not compatible with the standard alignment template:

detector.supports_alignment
# False
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

You can replace RetinaFace() with any of the supported detectors.

Output Format

All detectors return:

list[Face]

Each face contains:

for face in faces:
    bbox = face.bbox
    confidence = face.confidence
    landmarks = face.landmarks

The standard detectors return:

landmarks.shape == (5, 2)

with:

[left_eye,
 right_eye,
 nose,
 left_mouth,
 right_mouth]

BlazeFace returns:

landmarks.shape == (6, 2)
Visualization

UniFace provides a helper for visualizing detections:

from uniface.draw import draw_detections

draw_detections(
    image=image,
    faces=faces,
    vis_threshold=0.6,
)

cv2.imwrite("result.jpg", image)
Performance

You can benchmark detection performance on your own hardware:

python tools/detect.py --source image.jpg

Performance depends on:

CPU / GPU
Input resolution
Model variant
Confidence threshold
Number of detected faces
See Also
Recognition — Face embeddings and identification
Landmarks — Dense 106 / 98 / 68-point landmarks
Image Pipeline — Complete detection workflow
Threshold Calibration — Detection parameter tuning
CLI Tools — Command-line utilities

هذا هو **المحتوى فقط** بعد إزالة عناصر التوثيق الزائدة مثل `[svg]` وروابط GitHub المتكررة ووصف الصور، مع الحفاظ على المعلومات التقنية والجداول والأكواد.
