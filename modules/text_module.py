# ============================================================
# text_module.py
# Evidence-Aware Claim Verification Module
# ============================================================

# ----------------------------
# Core Imports
# ----------------------------
import torch
import faiss
import pickle
import numpy as np
import re

import spacy
import wikipediaapi
from wikidata.client import Client

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer


# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Paths (CHANGE ONLY IF NEEDED)
# ----------------------------
CLASSIFIER_PATH = "/content/drive/MyDrive/fake_news_system/classifier"
FAISS_GENERAL_DIR = "/content/drive/MyDrive/fake_news_system/faiss"
FAISS_PIB_PATH = "/content/drive/MyDrive/fake_news_system/faiss_pib.index"
FAISS_SC_PATH = "/content/drive/MyDrive/fake_news_system/faiss_sc.index"
PIB_CORPUS_PATH = "/content/drive/MyDrive/fake_news_system/pib_corpus.pkl"
SC_CORPUS_PATH = "/content/drive/MyDrive/fake_news_system/sc_corpus.pkl"


# ============================================================
# Load Models (FROZEN)
# ============================================================

# ---- Classifier ----
tokenizer_clf = AutoTokenizer.from_pretrained(
    CLASSIFIER_PATH,
    use_fast=False  # 🔒 stability fix for Colab / Python 3.12
)
model_clf = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_PATH
).to(DEVICE)
model_clf.eval()

# ---- General FAISS ----
index = faiss.read_index(f"{FAISS_GENERAL_DIR}/index.faiss")
with open(f"{FAISS_GENERAL_DIR}/corpus_texts.pkl", "rb") as f:
    corpus_texts = pickle.load(f)

# ---- Authoritative FAISS ----
faiss_pib = faiss.read_index(FAISS_PIB_PATH)
faiss_sc = faiss.read_index(FAISS_SC_PATH)

with open(PIB_CORPUS_PATH, "rb") as f:
    pib_corpus = pickle.load(f)

with open(SC_CORPUS_PATH, "rb") as f:
    sc_corpus = pickle.load(f)

# ---- Retriever ----
retriever = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=DEVICE
)

# ---- NLI ----
nli_tokenizer = AutoTokenizer.from_pretrained(
    "roberta-large-mnli",
    use_fast=False  # 🔒 stability fix for tokenizer JSON mismatch
)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli"
).to(DEVICE)
nli_model.eval()

# ---- NLP / Knowledge ----
nlp = spacy.load("en_core_web_sm")
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="FakeNewsDetection/1.0 (academic project)"
)
wikidata_client = Client()



# ============================================================
# Helper Functions
# ============================================================

@torch.no_grad()
def classifier_predict(claim, tokenizer, model, device):
    inputs = tokenizer(
        claim,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # label mapping: 0 = FAKE, 1 = REAL
    return float(probs[1]), probs


def extract_entities(text):
    doc = nlp(text)
    return list({
        ent.text
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "EVENT", "GPE", "WORK_OF_ART"}
    })


def link_entity_to_wikipedia(entity):
    page = wiki.page(entity)
    return page.title if page.exists() else None


def retrieve_evidence(claim, retriever_model, faiss_index, corpus, top_k=5):
    emb = retriever_model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    D, I = faiss_index.search(emb, top_k)

    return [
        {
            "text": corpus[idx],
            "similarity": float(score)
        }
        for score, idx in zip(D[0], I[0])
    ]


def aggregate_nli_weighted(nli_results):
    e = c = n = 0.0
    for r in nli_results:
        w = r["similarity"]
        e += w * r["entailment"]
        c += w * r["contradiction"]
        n += w * r["neutral"]

    total = e + c + n + 1e-8
    return {
        "entail_score": e / total,
        "contra_score": c / total,
        "neutral_score": n / total
    }


def nli_verify_evidence(claim, evidence):
    results = []
    for r in evidence:
        inputs = nli_tokenizer(
            r["text"],
            claim,
            return_tensors="pt",
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(
                nli_model(**inputs).logits, dim=-1
            )[0]

        results.append({
            "evidence": r["text"],
            "similarity": r["similarity"],
            "entailment": probs[2].item(),
            "contradiction": probs[0].item(),
            "neutral": probs[1].item()
        })

    return results


def is_verifiable_claim(claim):
    speculative = {
        "alien", "ufo", "time travel", "parallel universe",
        "teleport", "immortal", "conspiracy"
    }
    c = claim.lower()
    return not any(k in c for k in speculative)

def is_profession_claim(claim: str) -> bool:
    """
    Detects simple profession claims about well-known entities.
    Examples:
    - 'Virat Kohli is a cricketer.'
    - 'Sachin Tendulkar was a batsman.'
    """

    claim = claim.lower()

    # Common profession keywords (expandable)
    PROFESSION_KEYWORDS = {
        "cricketer", "footballer", "player", "athlete",
        "actor", "actress", "politician", "singer",
        "scientist", "author", "writer", "director",
        "businessman", "entrepreneur", "judge", "lawyer"
    }

    # Simple linguistic patterns
    PROFESSION_PATTERNS = [
        r".+ is a .+",
        r".+ is an .+",
        r".+ was a .+",
        r".+ was an .+",
        r".+ works as a .+"
    ]

    # Must contain a PERSON entity
    entities = extract_entities(claim)
    has_person = any(
        wiki.page(ent).exists() for ent in entities
    )

    if not has_person:
        return False

    # Check for profession keyword
    has_profession = any(
        prof in claim for prof in PROFESSION_KEYWORDS
    )

    if not has_profession:
        return False

    # Pattern match
    for p in PROFESSION_PATTERNS:
        if re.match(p, claim):
            return True

    return False



# ============================================================
# Authoritative Routing
# ============================================================

def route_claim(claim):
    c = claim.lower()

    if any(k in c for k in ["rbi", "repo rate", "monetary policy"]):
        return "RBI"
    if any(k in c for k in ["supreme court", "judgment", "verdict"]):
        return "SC"
    if any(k in c for k in ["government", "ministry", "policy", "budget"]):
        return "PIB"

    return "GENERAL"


def retrieve_from_authority(claim, index, corpus, top_k=5, min_sim=0.5):
    emb = retriever.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    D, I = index.search(emb, top_k)

    return [
        {
            "text": corpus[idx]["text"],
            "similarity": float(score),
            "authority": corpus[idx]["authority"]
        }
        for score, idx in zip(D[0], I[0])
        if score >= min_sim
    ]


def authoritative_gate(claim):
    route = route_claim(claim)

    if route == "SC":
        ev = retrieve_from_authority(claim, faiss_sc, sc_corpus)
        if not ev:
            return {
                "label": "FAKE",
                "confidence": 1.0,
                "reason": "No Supreme Court ruling confirms this claim"
            }

    if route == "PIB":
        ev = retrieve_from_authority(claim, faiss_pib, pib_corpus)
        if not ev:
            return {
                "label": "UNVERIFIABLE",
                "confidence": 1.0,
                "reason": "No official PIB confirmation"
            }

    return None


# ============================================================
# MAIN PUBLIC API
# ============================================================

def text_verify(claim: str):
    """
    Evidence-aware claim verification
    Returns: label, confidence, reason
    """

    # --------------------------------------------------
    # 0️⃣ Authoritative hard gate (RBI / SC / PIB)
    # --------------------------------------------------
    gate = authoritative_gate(claim)
    if gate:
        return gate

    # --------------------------------------------------
    # 1️⃣ Classifier prediction
    # --------------------------------------------------
    prob_real, _ = classifier_predict(
        claim, tokenizer_clf, model_clf, DEVICE
    )

    # --------------------------------------------------
    # 2️⃣ Speculative / unverifiable claims
    # --------------------------------------------------
    if not is_verifiable_claim(claim):
        return {
            "label": "UNVERIFIABLE",
            "confidence": round(1 - prob_real, 4),
            "reason": "Speculative or unverifiable claim"
        }

    # --------------------------------------------------
    # 3️⃣ Retrieve evidence
    # --------------------------------------------------
    retrieved = retrieve_evidence(
        claim,
        retriever,
        index,
        corpus_texts
    )

    # --------------------------------------------------
    # 4️⃣ No evidence fallback
    # --------------------------------------------------
    if not retrieved:
        # 🔒 SAFETY: protect well-known profession claims
        if is_profession_claim(claim) and prob_real >= 0.7:
            return {
                "label": "REAL",
                "confidence": round(prob_real, 4),
                "reason": "Well-known entity profession; classifier trusted"
            }

        return {
            "label": "REAL" if prob_real >= 0.5 else "FAKE",
            "confidence": round(prob_real, 4),
            "reason": "No strong evidence; classifier fallback"
        }

    # --------------------------------------------------
    # 5️⃣ NLI verification
    # --------------------------------------------------
    nli_results = nli_verify_evidence(claim, retrieved)
    scores = aggregate_nli_weighted(nli_results)

    # --------------------------------------------------
    # 6️⃣ 🚨 CONTRADICTION CHECK (with safety override)
    # --------------------------------------------------
    if scores["contra_score"] >= 0.7:

        # 🛡️ SAFETY OVERRIDE for profession claims
        if (
            is_profession_claim(claim)
            and prob_real >= 0.7
            and scores["contra_score"] < 0.85
        ):
            return {
                "label": "REAL",
                "confidence": round(prob_real, 4),
                "reason": "Well-known entity profession; classifier override"
            }

        return {
            "label": "FAKE",
            "confidence": round(scores["contra_score"], 4),
            "reason": "Strong contradictory evidence"
        }

    # --------------------------------------------------
    # 7️⃣ STRONG ENTAILMENT
    # --------------------------------------------------
    if scores["entail_score"] >= 0.7:
        return {
            "label": "REAL",
            "confidence": round(scores["entail_score"], 4),
            "reason": "Strong supporting evidence"
        }

    # --------------------------------------------------
    # 8️⃣ Weak or mixed evidence fallback
    # --------------------------------------------------
    # 🔒 SAFETY: profession claims again
    if is_profession_claim(claim) and prob_real >= 0.7:
        return {
            "label": "REAL",
            "confidence": round(prob_real, 4),
            "reason": "Well-known entity profession; classifier trusted"
        }

    return {
        "label": "REAL" if prob_real >= 0.5 else "FAKE",
        "confidence": round(prob_real, 4),
        "reason": "Weak or mixed evidence; classifier trusted"
    }

