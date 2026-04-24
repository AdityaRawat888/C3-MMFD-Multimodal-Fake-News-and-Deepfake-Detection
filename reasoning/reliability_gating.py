# ============================================================
# reliability_gating.py
# Confidence-aware modality reliability assessment
# ============================================================

def assess_reliability(
    text_out=None,
    av_out=None,
    text_conf_thresh=0.7,
    av_fake_thresh=0.6,
    av_real_thresh=0.45
):
    """
    Determines whether text and AV modalities are reliable
    enough to dominate the final decision.

    Parameters:
    - text_out: dict from text_module.text_verify() or None
    - av_out: dict from av_module.multimodal_fusion() or None

    Returns:
    - dict with reliability flags and explanations
    """

    reliability = {
        "text_reliable": False,
        "av_reliable": False,
        "notes": []
    }

    # --------------------------------------------------------
    # TEXT RELIABILITY
    # --------------------------------------------------------
    if text_out is not None:
        label = text_out.get("label")
        conf  = text_out.get("confidence", 0.0)

        if label in {"FAKE", "REAL"} and conf >= text_conf_thresh:
            reliability["text_reliable"] = True
            reliability["notes"].append(
                f"Text reliable (confidence={conf:.2f})"
            )
        else:
            reliability["notes"].append(
                f"Text not reliable (confidence={conf:.2f})"
            )

    # --------------------------------------------------------
    # AV RELIABILITY
    # --------------------------------------------------------
    if av_out is not None:
        label = av_out.get("label")
        score = av_out.get("final_spoof_score", None)

        if score is None:
            reliability["notes"].append("AV score missing")
        else:
            if label == "FAKE" and score >= av_fake_thresh:
                reliability["av_reliable"] = True
                reliability["notes"].append(
                    f"AV reliable FAKE (score={score:.2f})"
                )
            elif label == "REAL" and score <= av_real_thresh:
                reliability["av_reliable"] = True
                reliability["notes"].append(
                    f"AV reliable REAL (score={score:.2f})"
                )
            else:
                reliability["notes"].append(
                    f"AV not reliable (score={score:.2f}, label={label})"
                )

    reliability["notes"] = "; ".join(reliability["notes"])
    return reliability
