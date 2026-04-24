# ============================================================
# cross_modal_relevance.py
# Cross-Modal Relevance Detection for C³-MMFD
# ============================================================

from sentence_transformers import SentenceTransformer, util
import numpy as np

# ------------------------------------------------------------
# Load lightweight semantic model (fast + stable)
# ------------------------------------------------------------
_relevance_model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------
# Helper: normalize text
# ------------------------------------------------------------
def _clean_text(text: str) -> str:
    return " ".join(text.lower().split())


# ------------------------------------------------------------
# Core relevance check
# ------------------------------------------------------------
def check_cross_modal_relevance(
    text: str,
    video_transcript: str = None,
    threshold_related: float = 0.40,
    threshold_unrelated: float = 0.25
):
    """
    Determines whether text and video content are semantically related.

    Parameters:
    - text: textual claim
    - video_transcript: ASR transcript or captions from video
    - threshold_related: similarity above which content is RELATED
    - threshold_unrelated: similarity below which content is UNRELATED

    Returns:
    - dict with:
        - relevance_label: RELATED / UNRELATED / UNKNOWN
        - similarity_score
        - explanation
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    if not text:
        return {
            "relevance_label": "UNKNOWN",
            "similarity_score": None,
            "explanation": "No text provided"
        }

    if not video_transcript or len(video_transcript.strip()) < 5:
        return {
            "relevance_label": "UNKNOWN",
            "similarity_score": None,
            "explanation": "No reliable video transcript available"
        }

    # ----------------------------
    # Clean inputs
    # ----------------------------
    text_clean = _clean_text(text)
    transcript_clean = _clean_text(video_transcript)

    # ----------------------------
    # Encode embeddings
    # ----------------------------
    text_emb = _relevance_model.encode(
        text_clean, convert_to_tensor=True
    )

    video_emb = _relevance_model.encode(
        transcript_clean, convert_to_tensor=True
    )

    # ----------------------------
    # Compute similarity
    # ----------------------------
    similarity = util.cos_sim(text_emb, video_emb).item()
    similarity = float(np.clip(similarity, 0.0, 1.0))

    # ----------------------------
    # Decision logic (3-way)
    # ----------------------------
    if similarity >= threshold_related:
        label = "RELATED"
        explanation = "High semantic overlap between text and video content"

    elif similarity <= threshold_unrelated:
        label = "UNRELATED"
        explanation = "Low semantic similarity; content likely unrelated"

    else:
        label = "UNCERTAIN"
        explanation = "Moderate similarity; relationship unclear"

    return {
        "relevance_label": label,
        "similarity_score": round(similarity, 3),
        "explanation": explanation
    }
