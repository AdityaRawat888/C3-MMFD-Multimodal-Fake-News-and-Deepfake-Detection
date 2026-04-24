# ============================================================
# consistency_engine.py
# Cross-Modal Consistency & Contradiction Reasoning
# (Robust to missing modalities)
# ============================================================

def check_consistency(text_out=None, av_out=None):
    """
    Determines final decision based on available modalities.

    Parameters:
    - text_out: dict from text_module.text_verify() or None
    - av_out: dict from av_module.multimodal_fusion() or None

    Returns:
    - dict with:
        - final_label
        - consistency_type
        - trusted_modality
        - rationale
    """

    # --------------------------------------------------------
    # 0️⃣ No input at all (edge case)
    # --------------------------------------------------------
    if text_out is None and av_out is None:
        return {
            "final_label": "UNVERIFIABLE",
            "consistency_type": "NO_INPUT",
            "trusted_modality": "NONE",
            "rationale": "No input data provided"
        }

    # --------------------------------------------------------
    # 1️⃣ TEXT-ONLY CASE
    # --------------------------------------------------------
    if text_out is not None and av_out is None:
        return {
            "final_label": text_out.get("label", "UNVERIFIABLE"),
            "consistency_type": "TEXT_ONLY",
            "trusted_modality": "TEXT",
            "rationale": "Only textual input provided; decision based on text analysis"
        }

    # --------------------------------------------------------
    # 2️⃣ VIDEO/AUDIO-ONLY CASE
    # --------------------------------------------------------
    if av_out is not None and text_out is None:
        return {
            "final_label": av_out.get("label", "UNVERIFIABLE"),
            "consistency_type": "AV_ONLY",
            "trusted_modality": "AV",
            "rationale": "Only media input provided; decision based on audio-visual analysis"
        }

    # --------------------------------------------------------
    # 3️⃣ BOTH MODALITIES AVAILABLE
    # --------------------------------------------------------

    text_label = text_out.get("label")
    av_label   = av_out.get("label")

    # ----------------------------
    # Agreement
    # ----------------------------
    if text_label == av_label and text_label in {"FAKE", "REAL"}:
        return {
            "final_label": text_label,
            "consistency_type": "AGREEMENT",
            "trusted_modality": "TEXT+AV",
            "rationale": "Both text and media agree"
        }

    # ----------------------------
    # AV uncertain → trust text
    # ----------------------------
    if av_label == "UNCERTAIN" and text_label in {"FAKE", "REAL"}:
        return {
            "final_label": text_label,
            "consistency_type": "TEXT_DOMINANT",
            "trusted_modality": "TEXT",
            "rationale": "Media evidence inconclusive; text-based verification trusted"
        }

    # ----------------------------
    # Text unverifiable → trust AV
    # ----------------------------
    if text_label == "UNVERIFIABLE" and av_label in {"FAKE", "REAL"}:
        return {
            "final_label": av_label,
            "consistency_type": "AV_DOMINANT",
            "trusted_modality": "AV",
            "rationale": "Text unverifiable; media authenticity decisive"
        }

    # ----------------------------
    # Direct contradiction
    # ----------------------------
    if text_label == "REAL" and av_label == "FAKE":
        return {
            "final_label": "FAKE",
            "consistency_type": "CONTRADICTION",
            "trusted_modality": "AV",
            "rationale": "Media manipulation contradicts textual claim"
        }

    if text_label == "FAKE" and av_label == "REAL":
        return {
            "final_label": "FAKE",
            "consistency_type": "CONTRADICTION",
            "trusted_modality": "TEXT",
            "rationale": "Textual evidence contradicts apparently real media"
        }

    # ----------------------------
    # Both uncertain / unverifiable
    # ----------------------------
    if (
        text_label in {"UNVERIFIABLE"} and
        av_label in {"UNCERTAIN"}
    ):
        return {
            "final_label": "UNVERIFIABLE",
            "consistency_type": "NO_SIGNAL",
            "trusted_modality": "NONE",
            "rationale": "Neither modality provides sufficient evidence"
        }

    # ----------------------------
    # Safe fallback
    # ----------------------------
    return {
        "final_label": "UNVERIFIABLE",
        "consistency_type": "FALLBACK",
        "trusted_modality": "NONE",
        "rationale": "Unhandled case; conservative abstention"
    }
