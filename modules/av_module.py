# ============================================================
# av_module.py
# Audio + Video Deepfake Detection (Frozen Inference Module)
# ============================================================

import os
import sys
import json
import cv2
import torch
import torchaudio
import numpy as np
import subprocess
import tempfile
import soundfile as sf

from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from modules.xception_df import XceptionDeepFake


# ============================================================
# DEVICE (MUST BE DEFINED FIRST)
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#############################################################
## Face detector
#############################################################
face_detector = MTCNN(
    image_size=224,
    margin=20,
    keep_all=True,
    device=DEVICE
)


# ============================================================
# PATHS (CHANGE ONLY IF REQUIRED)
# ============================================================

AASIST_ROOT = "/content/drive/MyDrive/models/AASIST"
AASIST_CONF = f"{AASIST_ROOT}/config/AASIST.conf"
AASIST_WEIGHT = f"{AASIST_ROOT}/weights/AASIST.pth"


# Make model code importable
if AASIST_ROOT not in sys.path:
    sys.path.insert(0, AASIST_ROOT)



# ============================================================
# AUDIO MODEL (AASIST)
# ============================================================

from models.AASIST import Model as AASISTModel

with open(AASIST_CONF, "r") as f:
    cfg = json.load(f)

audio_model = AASISTModel(cfg["model_config"]).to(DEVICE)
state = torch.load(AASIST_WEIGHT, map_location=DEVICE)
audio_model.load_state_dict(state, strict=False)
audio_model.eval()



# ------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------

def fix_aasist_input(wav: torch.Tensor):
    """
    Ensure waveform shape is [B, T] for AASIST
    """
    # wav may be [1, T] or [1, 1, T]
    while wav.dim() > 2:
        wav = wav.squeeze(1)

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    return wav



def is_silent_audio(wav_path, energy_thresh=1e-4):
    wav, _ = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    energy = np.mean(wav ** 2)
    return energy < energy_thresh


@torch.no_grad()
def audio_infer(wav_path):
    if not wav_path or not os.path.exists(wav_path):
        return {
            "audio_spoof_score": None,
            "audio_label": "NO_AUDIO"
        }

    wav, sr = torchaudio.load(wav_path)
    wav = wav.to(DEVICE)

    # Mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # Shape fix
    wav = fix_aasist_input(wav)

    out = audio_model(wav)

    # 🔑 UNWRAP AASIST OUTPUT
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    probs = torch.softmax(logits, dim=1)
    score = probs[0, 1].item()

    return {
        "audio_spoof_score": round(score, 4),
        "audio_label": "FAKE" if score >= 0.5 else "REAL"
    }



# ============================================================
# VIDEO MODEL (XCEPTION – FF++)
# ============================================================

VIDEO_WEIGHTS = "/content/drive/MyDrive/xception_ffpp_2000.pt"

video_model = XceptionDeepFake().to(DEVICE)
video_model.load_state_dict(
    torch.load(VIDEO_WEIGHTS, map_location=DEVICE)
)
video_model.eval()



def is_valid_face(face, min_size=96, blur_thresh=120.0):
    """
    Filters faces that are too small or too blurry.
    """

    # ---- size check ----
    _, h, w = face.shape
    if min(h, w) < min_size:
        return False

    # ---- convert to numpy safely ----
    face_np = face.permute(1, 2, 0).detach().cpu().numpy()

    # normalize → uint8 (CRITICAL FIX)
    if face_np.max() <= 1.0:
        face_np = (face_np * 255).astype("uint8")
    else:
        face_np = face_np.astype("uint8")

    face_np = np.ascontiguousarray(face_np)

    # ---- blur detection ----
    gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return blur_score >= blur_thresh




@torch.no_grad()
def video_infer(
    video_path,
    max_faces=120,
    min_faces=10,
    frame_stride=3,
    blur_thresh=50.0,
    debug=False
):
    cap = cv2.VideoCapture(video_path)

    faces = []
    blur_scores = []
    frame_indices = []

    frame_idx = 0
    frames_seen = 0
    frames_with_face = 0

    while cap.isOpened() and len(faces) < max_faces:
        ret, frame = cap.read()
        if not ret:
            break

        frames_seen += 1
        frame_idx += 1

        if frame_idx % frame_stride != 0:
            continue

        bgr = frame
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        boxes, _ = face_detector.detect(pil_img)
        if boxes is None:
            continue

        frames_with_face += 1

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h, w, _ = bgr.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = bgr[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < 20:
                continue
            if blur_score > 180 and np.random.rand() > 0.85:
                continue

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb).resize((224, 224))

            face_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])(face_pil)

            faces.append(face_tensor)
            blur_scores.append(blur_score)
            frame_indices.append(frame_idx)

            if len(faces) >= max_faces:
                break

    cap.release()

    # ------------------------------------------------
    # LOW-QUALITY GUARD (CRITICAL FIX)
    # ------------------------------------------------

    low_quality = False

    if len(faces) < min_faces:
        if debug:
            print(
                f"[LOW QUALITY] faces={len(faces)} | "
                f"frames_seen={frames_seen} | "
                f"frames_with_face={frames_with_face}"
            )

        low_quality = True 



    batch = torch.stack(faces).to(DEVICE).float()

    # ------------------------------------------------
    # MODEL FORWARD
    # ------------------------------------------------
    logits = video_model(batch)
    probs = torch.sigmoid(logits.squeeze())

    # ------------------------------------------------
    # DEBUG: RAW PROBABILITY STATISTICS
    # ------------------------------------------------
    if debug:
        print("\n[RAW PROB STATS]")
        print("min   :", probs.min().item())
        print("mean  :", probs.mean().item())
        print("median:", torch.median(probs).item())
        print("q75   :", torch.quantile(probs, 0.75).item())
        print("q85   :", torch.quantile(probs, 0.85).item())
        print("q95   :", torch.quantile(probs, 0.95).item())
        print("q10   :", torch.quantile(probs, 0.10).item())

    # ------------------------------------------------
    # ROBUST DISTRIBUTION-AWARE AGGREGATION (FINAL)
    # ------------------------------------------------
    mean_p = probs.mean().item()
    median_p = torch.median(probs).item()
    std_p = probs.std().item()
    q95 = torch.quantile(probs, 0.95).item()
    q10 = torch.quantile(probs, 0.10).item()

    # Defaults
    score = 0.5
    label = "UNCERTAIN"



    # ==================================================
    # CASE 1️⃣ STRONG, CONSISTENT FAKE (HIGHEST PRIORITY)
    # ==================================================
    # Clean, stable, high-confidence manipulation
    if (
        median_p >= 0.7 and
        std_p <= 0.25 and 
        q10 >= 0.66
    ):
        print("🔥 CASE 1: STRONG FAKE")
        score = mean_p
        label = "FAKE"


    elif (
        q95 >= 0.95 and
        median_p >= 0.55 and
        std_p >= 0.25
    ):
        print("Weak Deepfake")
        score = q95
        label = "FAKE"



    # ==================================================
    # CASE 2️⃣ SPARSE / PARTIAL DEEPFAKE
    # ==================================================
    # Few frames extremely fake, others clean
    elif (
        q95 >= 0.85 and
        std_p >= 0.30 and
        median_p <= 0.6
    ):
        print("🔥 CASE 2: SPARSE FAKE")
        score = q95
        label = "FAKE"


    # ==================================================
    # CASE 3️⃣ HALLUCINATED FAKE ON REAL VIDEO
    # ==================================================
    # Model fires strongly but inconsistently
    elif (
        mean_p >= 0.7 and
        std_p >= 0.30 and
        median_p <= 0.6 and
        std_p > 0.25
    ):
        print("🔥 CASE 3: HALLUCINATED FAKE")
        score = 1 - mean_p
        label = "REAL"


    # ==================================================
    # CASE 4️⃣ STRONG, CONSISTENT REAL
    # ==================================================
    elif (
        mean_p <= 0.3 and
        median_p <= 0.3
    ):
        print("🔥 CASE 4: STRONG REAL")
        score = 1 - mean_p
        label = "REAL"


    # ==================================================
    # ELSE: UNCERTAIN
    # ==================================================
    else:
        score = 0.5
        label = "UNCERTAIN"




    # ------------------------------------------------
    # DEBUG OUTPUT
    # ------------------------------------------------
    if debug:
        print("\n================ VIDEO DEBUG =================")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Frames seen: {frames_seen}")
        print(f"Frames with faces: {frames_with_face}")
        print(f"Faces used: {len(faces)}")

        print("\n[Blur stats]")
        print(
            f"min={min(blur_scores):.1f}, "
            f"mean={np.mean(blur_scores):.1f}, "
            f"max={max(blur_scores):.1f}"
        )

        print("\n[Probability stats]")
        print(
            f"min={probs.min():.4f}, "
            f"mean={probs.mean():.4f}, "
            f"std={probs.std():.4f}, "
            f"max={probs.max():.4f}"
        )

        if debug:
            print("\n[DISTRIBUTION-AWARE AGGREGATION]")
            print(
                f"mean={mean_p:.4f}, "
                f"median={median_p:.4f}, "
                f"std={std_p:.4f}"
            )
            print(f"FINAL SCORE: {score:.4f} → {label}")
        print("================================================")

    return {
        "video_spoof_score": round(score, 4),
        "video_label": label,
        "frames_used": len(faces),
        "low_quality": low_quality 
    }






# ============================================================
# AUDIO EXTRACTION FROM VIDEO
# ============================================================

def has_audio_stream(video_path):
    cmd = [
        "ffprobe", "-loglevel", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() != ""


def extract_audio(video_path, sr=16000):
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "audio.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        wav_path
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return wav_path
    except:
        return None


# ============================================================
# FUSION LOGIC (ASYMMETRIC, AUDIO CONFIRMS ONLY)
# ============================================================

def fuse_scores(video_score, audio_score=None):
    """
    Audio can CONFIRM fake, but never override confident video.
    """

    # Confident video FAKE
    if video_score >= 0.65:
        return video_score

    # Confident video REAL
    if video_score <= 0.45:
        return video_score

    # Video UNCERTAIN → audio may help
    if audio_score is not None and audio_score >= 0.6:
        return audio_score

    # Still uncertain
    return video_score



# ============================================================
# MAIN PUBLIC API
# ============================================================

def multimodal_fusion(video_path):
    """
    Audio–Video deepfake inference with graceful degradation.
    Video is primary; audio may confirm fake when video is uncertain.
    """

    result = {}

    # ----------------------------
    # VIDEO INFERENCE
    # ----------------------------
    v_out = video_infer(video_path)
    result.update(v_out)

    video_score = v_out["video_spoof_score"]
    video_label = v_out["video_label"]
    low_quality = v_out.get("low_quality", False)

    # ----------------------------
    # AUDIO INFERENCE (OPTIONAL)
    # ----------------------------
    audio_score = None
    audio_used = False

    if has_audio_stream(video_path):
        wav_path = extract_audio(video_path)
        if wav_path and not is_silent_audio(wav_path):
            a_out = audio_infer(wav_path)
            audio_score = a_out["audio_spoof_score"]
            audio_used = True
            result.update(a_out)

    # ----------------------------
    # ASYMMETRIC FUSION LOGIC
    # ----------------------------
    # 1️⃣ Trust confident video
    if video_label != "UNCERTAIN":
        final_score = video_score

    # 2️⃣ Video uncertain → audio may confirm FAKE
    elif audio_score is not None and audio_score >= 0.6:
        final_score = audio_score

    # 3️⃣ Still uncertain → fall back to video
    else:
        final_score = video_score

    # ----------------------------
    # FINAL LABEL
    # ----------------------------
    if final_score >= 0.6:
        final_label = "FAKE"
    elif final_score <= 0.45:
        final_label = "REAL"
    else:
        final_label = "UNCERTAIN"

    # ----------------------------
    # OUTPUT
    # ----------------------------
    result.update({
        "final_spoof_score": round(final_score, 4),
        "label": final_label,
        "audio_used": audio_used,
        "low_quality_video": low_quality,
        "modalities_used": (
            ["video"] + (["audio"] if audio_used else [])
        )
    })

    return result




