import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate
import torch

# ── 1. Label setup ────────────────────────────────────────────────────────────
LABELS = ["O", "B-INSURED", "I-INSURED", "B-COVERAGE", "I-COVERAGE",
          "B-PREMIUM", "I-PREMIUM", "B-POLICY_DATE", "I-POLICY_DATE",
          "B-EXCLUSION", "I-EXCLUSION", "B-POLICY_LIMIT", "I-POLICY_LIMIT"]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
MODEL_NAME = "bert-base-uncased"

# ── 2. Load labeled data ───────────────────────────────────────────────────────
def load_data(path="data/labeled_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── 3. Convert character-level spans → token-level BIO tags ───────────────────
def char_to_token_labels(example, tokenizer):
    text = example["text"]
    entities = example["entities"]

    char_labels = ["O"] * len(text)
    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        for i in range(start, min(end, len(text))):
            if i == start:
                char_labels[i] = "B-" + label
            else:
                char_labels[i] = "I-" + label

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_offsets_mapping=True
    )

    token_labels = []
    for (tok_start, tok_end) in encoding["offset_mapping"]:
        if tok_start == tok_end:          # special token [CLS], [SEP], [PAD]
            token_labels.append(-100)
        else:
            token_labels.append(LABEL2ID[char_labels[tok_start]])

    encoding["labels"] = token_labels
    encoding.pop("offset_mapping")
    return encoding

# ── 4. Build HuggingFace Dataset ───────────────────────────────────────────────
def build_dataset(raw_data, tokenizer):
    processed = []
    for example in raw_data:
        try:
            enc = char_to_token_labels(example, tokenizer)
            processed.append(enc)
        except Exception as e:
            print(f"Skipping example due to error: {e}")
    return Dataset.from_list(processed)

# ── 5. Compute F1 per entity class (seqeval) ──────────────────────────────────
seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, labels):
        true_row, pred_row_out = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_row.append(ID2LABEL[l])
            pred_row_out.append(ID2LABEL[p])
        true_labels.append(true_row)
        true_preds.append(pred_row_out)

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    output = {"overall_f1": results["overall_f1"],
               "overall_precision": results["overall_precision"],
               "overall_recall": results["overall_recall"]}
    for entity in ["INSURED","COVERAGE","PREMIUM","POLICY_DATE","EXCLUSION","POLICY_LIMIT"]:
        if entity in results:
            output[f"{entity}_f1"] = results[entity]["f1"]
    return output

# ── 6. Train ──────────────────────────────────────────────────────────────────
def train():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    print("Loading and processing data...")
    raw_data = load_data()

    # 80/20 train/eval split
    split = int(0.8 * len(raw_data))
    train_data = build_dataset(raw_data[:split], tokenizer)
    eval_data  = build_dataset(raw_data[split:], tokenizer)
    print(f"Train: {len(train_data)} samples | Eval: {len(eval_data)} samples")

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="models/bert-insurance-ner",
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        logging_steps=5,
        warmup_steps=5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving best model...")
    trainer.save_model("models/bert-insurance-ner/best")
    tokenizer.save_pretrained("models/bert-insurance-ner/best")
    print("Model saved to models/bert-insurance-ner/best")

    print("\n── Final evaluation metrics ──")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if "f1" in k or "precision" in k or "recall" in k:
            print(f"  {k}: {v:.4f}")

    return metrics

if __name__ == "__main__":
    metrics = train()