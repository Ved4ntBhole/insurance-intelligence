import pdfplumber
import os
import json
import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def extract_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def parse_pdf(pdf_path):
    result = {
        "filename": os.path.basename(pdf_path),
        "full_text": "",
        "pages": [],
        "sentences": []
    }
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    cleaned = clean_text(page_text)
                    all_text.append(cleaned)
                    result["pages"].append({
                        "page_number": i + 1,
                        "text": cleaned
                    })
            result["full_text"] = "\n\n".join(all_text)
            result["sentences"] = extract_sentences(result["full_text"])
    except Exception as e:
        print(f"Error parsing {pdf_path}: {e}")
    return result

def parse_all_pdfs(data_folder="data"):
    all_docs = []
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs")
    for pdf_file in pdf_files:
        path = os.path.join(data_folder, pdf_file)
        print(f"Parsing: {pdf_file}")
        doc = parse_pdf(path)
        all_docs.append(doc)
        print(f"  -> {len(doc['sentences'])} sentences extracted")
    return all_docs

def save_parsed_docs(docs, output_path="data/parsed_docs.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(docs)} documents to {output_path}")

if __name__ == "__main__":
    docs = parse_all_pdfs("data")
    save_parsed_docs(docs)
    print("\n--- Sample sentences from first document ---")
    if docs:
        for s in docs[0]["sentences"][:10]:
            print(f"  • {s}")