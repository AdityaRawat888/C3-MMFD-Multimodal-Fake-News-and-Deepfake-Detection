# ============================================================
# decision_layer.py
# Final Decision Layer for C³-MMFD (Text + AV + Video)
# ============================================================

from c3_mmfd.consistency_engine import check_consistency
from c3_mmfd.reliability_gating import assess_reliability


def final_decision(text_out=None, av_out=None, video_out=None):

    # --------------------------------------------------------
    # 1️⃣ Get consistency-based decision (Text + AV)
    # --------------------------------------------------------
    consistency = check_consistency(text_out=text_out, av_out=av_out)

    final_label = consistency["final_label"]
    decision_type = consistency["consistency_type"]
    trusted_modality = consistency["trusted_modality"]

    # --------------------------------------------------------
    # 2️⃣ Assess reliability
    # --------------------------------------------------------
    reliability = assess_reliability(text_out=text_out, av_out=av_out)

    video_reliable = (
        video_out is not None
        and video_out.get("label") in {"REAL", "FAKE"}
        and video_out.get("confidence", 0) >= 0.7
    )
    reliability["video_reliable"] = video_reliable

    # --------------------------------------------------------
    # 🎥 Video Deepfake Arbitration
    # --------------------------------------------------------
    if video_out and video_reliable and final_label in {"REAL", "FAKE"}:

        video_label = video_out["label"]

        if decision_type == "CONFLICT":
            final_label = video_label
            decision_type = "VIDEO_ARBITRATION"
            trusted_modality = "VIDEO"

        elif video_label != final_label and video_out["confidence"] >= 0.85:
            final_label = "UNCERTAIN"
            decision_type = "VIDEO_CONTRADICTION"
            trusted_modality = "NONE"

    # --------------------------------------------------------
    # 3️⃣ Reliability-aware overrides
    # --------------------------------------------------------
    if trusted_modality == "TEXT" and not reliability["text_reliable"]:
        final_label = "UNCERTAIN"
        decision_type = "TEXT_UNRELIABLE"
        trusted_modality = "NONE"

    if trusted_modality == "AV" and not reliability["av_reliable"]:
        final_label = "UNCERTAIN"
        decision_type = "AV_UNRELIABLE"
        trusted_modality = "NONE"

    if trusted_modality == "VIDEO" and not reliability["video_reliable"]:
        final_label = "UNCERTAIN"
        decision_type = "VIDEO_UNRELIABLE"
        trusted_modality = "NONE"

    if (
        not reliability["text_reliable"]
        and not reliability["av_reliable"]
        and not reliability["video_reliable"]
        and final_label in {"REAL", "FAKE"}
    ):
        final_label = "UNCERTAIN"
        decision_type = "LOW_RELIABILITY"
        trusted_modality = "NONE"

    # --------------------------------------------------------
    # 4️⃣ Explanation
    # --------------------------------------------------------
    explanation = (
        f"Decision based on {decision_type}. "
        f"Trusted modality: {trusted_modality}. "
        f"Text reliable: {reliability['text_reliable']}, "
        f"AV reliable: {reliability['av_reliable']}, "
        f"Video reliable: {reliability['video_reliable']}. "
        f"{reliability['notes']}"
    )

    # --------------------------------------------------------
    # 5️⃣ Final output
    # --------------------------------------------------------
    return {
        "final_label": final_label,
        "decision_type": decision_type,
        "trusted_modality": trusted_modality,
        "explanation": explanation,
        "consistency": consistency,
        "reliability": reliability
    }
