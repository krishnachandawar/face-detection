from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = BASE_DIR / "haarcascade_eye.xml"

FACE_SIZE = 200
DETECT_SCALE = 1.05
DETECT_NEIGHBORS = 8
DETECT_MIN_SIZE = (120, 120)
BLUR_MIN_VARIANCE = 50.0

LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
CONFIDENCE_THRESHOLD = 42.0

_clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
_face_cascade = None
_eye_cascade = None


def _load_cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is None:
        if not CASCADE_PATH.exists():
            raise FileNotFoundError(f"Missing cascade file: {CASCADE_PATH}")
        _face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if _face_cascade.empty():
            raise RuntimeError("Failed to load face cascade.")
    if _eye_cascade is None and EYE_CASCADE_PATH.exists():
        _eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
        if _eye_cascade.empty():
            _eye_cascade = None
    return _face_cascade, _eye_cascade


def enhance_gray(gray):
    return _clahe.apply(gray)


def detect_faces(gray):
    face_cascade, _ = _load_cascades()
    faces = face_cascade.detectMultiScale(
        enhance_gray(gray),
        scaleFactor=DETECT_SCALE,
        minNeighbors=DETECT_NEIGHBORS,
        minSize=DETECT_MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces


def largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: int(box[2]) * int(box[3]))


def is_sharp(face_gray):
    return cv2.Laplacian(face_gray, cv2.CV_64F).var() >= BLUR_MIN_VARIANCE


def _align_with_eyes(face_gray):
    _, eye_cascade = _load_cascades()
    if eye_cascade is None:
        return face_gray

    h, w = face_gray.shape[:2]
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(max(12, w // 10), max(12, h // 10)),
    )
    if len(eyes) < 2:
        return face_gray

    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    centers = [(ex + ew / 2.0, ey + eh / 2.0) for ex, ey, ew, eh in eyes]
    if centers[0][0] > centers[1][0]:
        centers[0], centers[1] = centers[1], centers[0]

    dx = centers[1][0] - centers[0][0]
    dy = centers[1][1] - centers[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) > 25:
        return face_gray

    eyes_center = ((centers[0][0] + centers[1][0]) / 2.0, (centers[0][1] + centers[1][1]) / 2.0)
    matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    return cv2.warpAffine(face_gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_face(gray, box, margin=0.18):
    x, y, w, h = [int(v) for v in box]
    extra_w = int(w * margin)
    extra_h = int(h * margin)
    x1 = max(0, x - extra_w)
    y1 = max(0, y - extra_h)
    x2 = min(gray.shape[1], x + w + extra_w)
    y2 = min(gray.shape[0], y + h + extra_h)
    return gray[y1:y2, x1:x2]


def preprocess_face(gray, box=None):
    face = crop_face(gray, box) if box is not None else gray
    if face.size == 0:
        return None
    face = _align_with_eyes(face)
    face = enhance_gray(face)
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
    return face


def augment_faces(face):
    samples = [face]
    h, w = face.shape[:2]
    center = (w / 2.0, h / 2.0)
    for angle in (-8, -4, 4, 8):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        samples.append(cv2.warpAffine(face, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
    for delta in (-35, -18, 18, 35):
        samples.append(cv2.convertScaleAbs(face, alpha=1.0, beta=delta))
    noise = np.random.normal(0, 6, face.shape).astype(np.int16)
    noisy = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    samples.append(noisy)
    return samples


def create_recognizer():
    return cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y,
    )
