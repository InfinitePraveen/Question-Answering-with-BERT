# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
- Considering SQuAD 2.0 support (unanswerable questions)
- Considering a Dockerfile for the web app

## [1.0.0] - 2026-09-20
### Added
- `1_Data_Exploration.ipynb` — loads and explores the SQuAD dataset
- `2_Fine_Tune_BERT_SQuAD.ipynb` — fine-tunes DistilBERT on a SQuAD subset (CPU-friendly)
- `3_Model_Evaluation.ipynb` — Exact Match / F1 evaluation + sample predictions
- Flask web app (`app.py`) with a simple context + question form
- Fallback to `distilbert-base-cased-distilled-squad` from the HuggingFace Hub when no locally fine-tuned model is present
- README, CONTRIBUTING, LICENSE

## [0.1.0] - 2026-09-20
### Added
- Initial project scaffold and repo structure
- requirements.txt with core dependencies
