# C³-MMFD: Reliability-Aware Multimodal Fake News and Deepfake Detection

## Overview

C³-MMFD (Credibility-aware Cross-modal Multimodal Fake News Detection) is a multimodal artificial intelligence framework designed to detect misinformation across multiple modalities including text, images, and videos.

With the rapid growth of misinformation on social media, fake news and deepfake media have become a major challenge. Traditional detection systems often analyze a single modality independently and assume equal reliability across all modalities.

C³-MMFD introduces a reliability-aware multimodal fusion framework that evaluates and combines information from multiple modalities to improve detection accuracy and robustness.

This repository contains the implementation, experiments, and evaluation of the proposed framework.

---

## Key Features

• Multimodal fake news detection  
• Deepfake media detection  
• Reliability-aware modality fusion  
• Cross-modal feature extraction  
• Deep learning based classification  
• Experimental evaluation on benchmark datasets  

---

## System Architecture

The proposed framework integrates information from multiple modalities and evaluates their reliability before performing final classification.

Architecture Diagram:


![Architecture](docs/architecture.png)
---

## Datasets Used

The framework is evaluated on multiple benchmark datasets commonly used in misinformation detection research.

| Dataset | Description |
|------|------|
| **LIAR** | Short political statements labeled for truthfulness |
| **FEVER** | Fact extraction and verification dataset |
| **SciFact** | Scientific claim verification dataset |

Dataset links are provided in the `datasets` directory.

---

## Project Structure
