# ============================================================
# face_utils.py
# Face extraction utilities for video deepfake detection
# ============================================================

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe face detector
mp_face_detection = mp.solutions.face_detection


def extract_face_frames(
    video_path,
    max_faces=10,
    frame_stride=5,
    target_size=(224, 224)
):
    """
    Extract face crops from a video using MediaPipe.

    Parameters:
    - video_path (str): path to video file
    - max_faces (int): maximum number of face crops to extract
    - frame_stride (int): sample every Nth frame
    - target_size (tuple): resize faces to (H, W)

    Returns:
    - List[np.ndarray]: list of face images (H, W, 3), BGR format
    """

    cap = cv2.VideoCapture(video_path)
    faces = []

    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return faces

    frame_idx = 0

    with mp_face_detection.FaceDetection(
        model_selection=1,               # long-range model
        min_detection_confidence=0.5
    ) as face_detector:

        while cap.isOpened() and len(faces) < max_faces:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Sample frames
            if frame_idx % frame_stride != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if not results.detections:
                continue

            h, w, _ = frame.shape

            for det in results.detections:
                bbox = det.location_data.relative_bounding_box

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Clamp to frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.resize(face, target_size)
                faces.append(face)

                if len(faces) >= max_faces:
                    break

    cap.release()
    return faces
