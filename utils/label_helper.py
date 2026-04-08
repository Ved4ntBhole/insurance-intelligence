import json
import os

def load_parsed_docs(path="data/parsed_docs.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def show_sentences_for_labeling(docs, start=0, count=30):
    all_sentences = []
    for doc in docs:
        for s in doc["sentences"]:
            all_sentences.append({
                "text": s,
                "source": doc["filename"]
            })
    batch = all_sentences[start:start+count]
    for i, item in enumerate(batch):
        print(f"\n[{start+i}] {item['text'][:200]}")
        print(f"     (from: {item['source']})")
    return all_sentences

if __name__ == "__main__":
    docs = load_parsed_docs()
    sentences = show_sentences_for_labeling(docs, start=0, count=20)
    print(f"\nTotal sentences available: {len(sentences)}")