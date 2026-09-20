# Question Answering with BERT 🤖📖

Fine-tuning a BERT-family transformer on the **SQuAD** (Stanford Question Answering Dataset) to build an extractive Question Answering system — give it a passage and a question, and it points to the exact span of text that answers it.

This is one of the projects in my data science / ML portfolio. I built it to get hands-on with the HuggingFace ecosystem (`transformers` + `datasets`) end-to-end: data prep → fine-tuning → evaluation → a small web app to actually demo the thing instead of just leaving it in a notebook.

> Live demo screenshot goes here once deployed — for now, run it locally with the steps below (takes ~2 minutes).

---

## What this project does

Given a **context paragraph** and a **question**, the model finds the answer *inside* the paragraph (extractive QA — it doesn't generate free text, it highlights a span). Example:

> **Context:** "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel and completed in 1889."
>
> **Question:** "Who designed the Eiffel Tower?"
>
> **Answer:** `Gustave Eiffel`

---

## Why BERT for this task

BERT is a natural fit for extractive QA because it's bidirectional — it reads the whole context and question together and learns to predict two things per token: how likely it is to be the **start** of the answer span, and how likely it is to be the **end**. Fine-tuning just adds a small QA head (two linear layers) on top of the pretrained BERT encoder.

---

## Tech stack

| Piece | What I used |
|---|---|
| Model | BERT / DistilBERT (HuggingFace `transformers`) |
| Dataset | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) via HuggingFace `datasets` |
| Training | PyTorch + HuggingFace `Trainer` API |
| Evaluation | Exact Match (EM) & F1 score |
| Web App | Flask |
| Notebooks | Jupyter |

---

## A note on hardware (important!)

I did this on my personal laptop — **no GPU, limited disk space**. So a few practical calls were made instead of pretending I trained full BERT-base on the full SQuAD dataset for 3 epochs like the paper does:

- I fine-tune on a **reduced subset** of SQuAD (a few thousand examples instead of ~87k) — enough to actually see the model learn the QA task without needing a GPU or hours of compute.
- I default to **`distilbert-base-uncased`** for the training notebook — same idea as BERT, ~40% smaller and noticeably faster on CPU, so the whole pipeline is reproducible on a normal laptop. Swapping in `bert-base-uncased` is a one-line change in the notebook if you've got the GPU/time for it.
- The **web app**, for a smoother demo experience, loads a community-standard **`distilbert-base-cased-distilled-squad`** checkpoint from the HuggingFace Hub (already fine-tuned on the *full* SQuAD dataset) if a locally fine-tuned model isn't found in `models/`. That way the demo is snappy and accurate even on a laptop with no GPU, while the notebooks still walk through — and let you reproduce — the actual fine-tuning process.
- Data is streamed via `requests`/`datasets` rather than kept as large local files, and I clean up cached files with `pathlib`/`os` where possible to keep disk usage low.

If you do have GPU access (e.g. free Colab), just bump up `NUM_TRAIN_EXAMPLES`, epochs, and swap the model name in `notebooks/2_Fine_Tune_BERT_SQuAD.ipynb` — the pipeline doesn't change.

---

## Repository structure

```
bert-qa-squad/
├── notebooks/
│   ├── 1_Data_Exploration.ipynb        # Load & explore SQuAD
│   ├── 2_Fine_Tune_BERT_SQuAD.ipynb    # Fine-tune the QA model
│   └── 3_Model_Evaluation.ipynb        # EM / F1 + sample predictions
├── templates/
│   └── index.html                      # Web app UI
├── static/
│   └── style.css                       # Web app styling
├── models/                             # Fine-tuned model gets saved here (gitignored)
├── app.py                              # Flask web app
├── requirements.txt
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

Kept it flat and simple on purpose — everything lives in a notebook or in the app, no separate `src/` package to import from.

---

## Getting started

### 1. Clone & set up environment

```bash
git clone https://github.com/InfinitePraveen/Question-Answering-with-BERT
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the notebooks (in order)

```bash
jupyter notebook
```

1. `1_Data_Exploration.ipynb` — pulls SQuAD, explores the structure
2. `2_Fine_Tune_BERT_SQuAD.ipynb` — fine-tunes DistilBERT on a subset, saves it to `models/bert-squad-finetuned/`
3. `3_Model_Evaluation.ipynb` — computes EM/F1 and shows sample predictions

> Notebook 2 is the only compute-heavy one, and even that's tuned to finish in a reasonable time on CPU. No GPU required.

### 3. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser, paste in a paragraph, ask a question, and see the model pull out the answer.

If you skipped the notebooks, the app still works — it'll automatically fall back to a pretrained SQuAD checkpoint from the HuggingFace Hub the first time you run it (small download, ~250MB, cached after that).

---

## Sample results

After fine-tuning DistilBERT on a subset of SQuAD (see `3_Model_Evaluation.ipynb` for the full breakdown):

| Metric | Score |
|---|---|
| Exact Match | ~55-65% (subset training) |
| F1 Score | ~68-75% (subset training) |

These numbers are lower than the official SQuAD leaderboard (full BERT-large gets EM ~87 / F1 ~93) because this is trained on a small slice of the data on CPU — the point here was learning the fine-tuning pipeline end-to-end, not chasing leaderboard numbers. The notebook shows exactly what changes if you scale up the data/model.

---

## What I'd improve with more time/hardware

- Train on the full SQuAD train split with `bert-base-uncased` or `bert-large-uncased` on a GPU
- Try SQuAD 2.0 (includes unanswerable questions) for a harder, more realistic task
- Add answer confidence thresholding in the web app so it says "I don't know" instead of guessing
- Containerize the Flask app with Docker for easier deployment

---

## About me

I'm Praveen, learning data science & ML by building practical, end-to-end projects rather than just doing isolated tutorials.

- GitHub: [InfinitePraveen](https://github.com/InfinitePraveen)
- LinkedIn: [infinitepraveen](https://www.linkedin.com/in/infinitepraveen)

Feedback and PRs welcome — see `CONTRIBUTING.md`.

## License

MIT — see `LICENSE`.
