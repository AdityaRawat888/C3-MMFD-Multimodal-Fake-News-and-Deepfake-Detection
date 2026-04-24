import torch
import numpy as np

from torchvision import transforms
from modules.xception_model import XceptionDeepfake



# ----------------------------
# CONFIG
# ----------------------------
# ----------------------------
# DEVICE
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD TRAINED DEEPFAKE MODEL
# ----------------------------
MODEL_PATH = "/content/drive/MyDrive/xception_ffpp_2000.pt"  # your trained checkpoint

video_model = XceptionDeepfake().to(DEVICE)
video_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE),
    strict=True
)
video_model.eval()



# ----------------------------
# TRANSFORMS (same as training)
# ----------------------------
val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ----------------------------
# MAIN API
# ----------------------------
def video_deepfake_score(video_path, num_frames=10):
    """
    Returns:
    {
        "label": "REAL" | "FAKE",
        "confidence": float,
        "prob_fake": float
    }
    """

    faces = extract_face_frames(video_path, num_frames=num_frames)

    if len(faces) == 0:
        return {
            "label": "UNCERTAIN",
            "confidence": 0.0,
            "prob_fake": None
        }

    probs = []

    with torch.no_grad():
        for face in faces:
            x = val_tfms(face).unsqueeze(0).to(DEVICE)
            logit = _model(x)
            prob_fake = torch.sigmoid(logit).item()
            probs.append(prob_fake)

    video_prob = float(np.mean(probs))

    label = "FAKE" if video_prob > 0.5 else "REAL"
    confidence = video_prob if label == "FAKE" else 1 - video_prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_fake": round(video_prob, 4)
    }
