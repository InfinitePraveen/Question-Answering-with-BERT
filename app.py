"""
Question Answering with BERT - Flask demo app.

Loads a fine-tuned QA model from ./models/bert-squad-finetuned if it exists
(produced by notebooks/2_Fine_Tune_BERT_SQuAD.ipynb). If it's not there,
falls back to a pretrained SQuAD checkpoint from the HuggingFace Hub so the
demo still works out of the box on a machine with no GPU.

Author: Praveen (https://github.com/InfinitePraveen)
"""

from pathlib import Path

import torch
from flask import Flask, render_template, request
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

app = Flask(__name__)

LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "bert-squad-finetuned"
FALLBACK_MODEL = "distilbert-base-cased-distilled-squad"

print("Preparing question-answering model, this can take a moment on first run...")

if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
    MODEL_SOURCE = str(LOCAL_MODEL_DIR)
    print(f"Using locally fine-tuned model from {MODEL_SOURCE}")
else:
    MODEL_SOURCE = FALLBACK_MODEL
    print(
        f"No local fine-tuned model found in {LOCAL_MODEL_DIR}.\n"
        f"Falling back to pretrained '{FALLBACK_MODEL}' from the HuggingFace Hub."
    )

qa_model = None
tokenizer = None


def get_qa_model_and_tokenizer():
    """Load the QA model + tokenizer lazily without relying on the missing pipeline task."""
    global qa_model, tokenizer

    if qa_model is None or tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
            qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_SOURCE)
            qa_model.eval()
        except Exception as exc:  # pragma: no cover - surfaced to the UI for debugging
            raise RuntimeError(
                f"Unable to load the question-answering model from '{MODEL_SOURCE}'. "
                f"Details: {exc}"
            ) from exc

    return qa_model, tokenizer


def answer_question(question: str, context: str):
    """Generate an answer span using the explicit QA model API."""
    model, tokenizer_instance = get_qa_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer_instance(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    start_idx = int(torch.argmax(start_logits).item())
    end_idx = int(torch.argmax(end_logits).item())

    if end_idx < start_idx:
        end_idx = start_idx

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer_instance.decode(answer_ids, skip_special_tokens=True).strip()

    start_prob = torch.softmax(outputs.start_logits, dim=-1)[0, start_idx].item()
    end_prob = torch.softmax(outputs.end_logits, dim=-1)[0, end_idx].item()
    confidence = round((start_prob * end_prob) * 100, 2)

    return answer, confidence

# A couple of example passages so the form isn't blank on first load
SAMPLE_CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
    "in Paris, France. It was designed by Gustave Eiffel and built between "
    "1887 and 1889 as the entrance arch to the 1889 World's Fair. It was "
    "initially criticized by some of France's leading artists and "
    "intellectuals for its design, but it has become a global cultural icon "
    "of France and one of the most recognizable structures in the world."
)
SAMPLE_QUESTION = "Who designed the Eiffel Tower?"


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        context=SAMPLE_CONTEXT,
        question=SAMPLE_QUESTION,
        answer=None,
        score=None,
        model_source=MODEL_SOURCE,
    )


@app.route("/predict", methods=["POST"])
def predict():
    context = request.form.get("context", "").strip()
    question = request.form.get("question", "").strip()

    if not context or not question:
        return render_template(
            "index.html",
            context=context,
            question=question,
            answer=None,
            score=None,
            error="Please provide both a passage and a question.",
            model_source=MODEL_SOURCE,
        )

    try:
        answer, score = answer_question(question, context)
    except Exception as exc:
        return render_template(
            "index.html",
            context=context,
            question=question,
            answer=None,
            score=None,
            error=str(exc),
            model_source=MODEL_SOURCE,
        )

    return render_template(
        "index.html",
        context=context,
        question=question,
        answer=answer,
        score=score,
        model_source=MODEL_SOURCE,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
