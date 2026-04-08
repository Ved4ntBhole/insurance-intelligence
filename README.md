# Insurance Document Intelligence System

An end-to-end NLP pipeline that ingests insurance policy PDFs, extracts key entities using fine-tuned BERT, detects missing clauses via sentence-transformer cosine similarity, and presents results through an interactive Streamlit dashboard.

## Problem Statement

Insurance underwriters process hundreds of policy documents manually to check coverage completeness. This system automates entity extraction and clause verification, flagging anomalies for human review.

## Architecture

PDF Upload → Text Extraction (pdfplumber)
→ BERT NER → Entity Table + Highlighted Text
→ Sentence-BERT Similarity → Clause Coverage + Anomaly Report
→ Streamlit Dashboard

## Models

| Component | Model | Purpose |
|---|---|---|
| Named Entity Recognition | bert-base-uncased (fine-tuned) | Extract 6 entity types from policy text |
| Clause Similarity | all-MiniLM-L6-v2 | Match document clauses against 12 standard templates |

## Entity Types

| Entity | Description | Example |
|---|---|---|
| INSURED | Name of insured party | "Tata Motors Limited" |
| COVERAGE | What is covered | "Fire", "Flood", "Earthquake" |
| PREMIUM | Amount paid | "Rs. 18,750" |
| POLICY_DATE | Policy dates | "01/04/2024" |
| EXCLUSION | What is not covered | "spontaneous combustion" |
| POLICY_LIMIT | Maximum payout | "Rs. 50,00,000" |

## Evaluation Metrics

We use **F1-score per entity class** (not accuracy) because:

- The dataset is heavily imbalanced — 'O' tokens dominate
- Accuracy would be misleadingly high (~85%+)
- A missed EXCLUSION or POLICY_LIMIT is a high-stakes error
- F1 balances precision and recall where both matter

| Metric | Score |
|---|---|
| Overall F1 | 0.3889 |
| Overall Precision | 0.3684 |
| Overall Recall | 0.4118 |
| POLICY_DATE F1 | 1.0000 |
| EXCLUSION F1 | 0.4000 |
| INSURED F1 | 0.2857 |
| POLICY_LIMIT F1 | 0.2857 |

COVERAGE and PREMIUM scored 0.0 on this eval split due to data distribution — neither entity appeared in the 12 evaluation sentences. This is a **data problem, not a model problem**.

## Clause Similarity

12 standard insurance clauses are checked per document using cosine similarity on sentence embeddings. Threshold: **0.45**.

- Score ≥ 0.45 → PRESENT
- Score 0.30–0.45 → possible paraphrasing, manual review recommended
- Score < 0.30 → likely absent

## Limitations

- 60 labeled training samples — production needs 500+
- Scanned/image PDFs not supported (requires OCR)
- Threshold tuned on Standard Fire & Special Perils documents
- No support for tables inside PDFs yet

## Setup

```bash
conda create -n insurance-nlp python=3.10 -y
conda activate insurance-nlp
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

insurance-intelligence/
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
├── data/
│   ├── *.pdf               # Insurance PDFs
│   ├── parsed_docs.json    # Extracted text
│   └── labeled_data.json   # 60 labeled NER samples
├── models/
│   └── bert-insurance-ner/
│       └── best/           # Fine-tuned BERT model
└── utils/
├── pdf_parser.py       # PDF text extraction
├── similarity.py       # Clause similarity checker
├── train_ner.py        # BERT fine-tuning script
└── evaluate.py         # Evaluation report