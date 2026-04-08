from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Standard clauses every good insurance policy should have
STANDARD_CLAUSES = {
    "fire_coverage": "Loss or damage caused by fire is covered under this policy.",
    "flood_coverage": "Flood and inundation damage is covered under this policy.",
    "earthquake_coverage": "Earthquake damage including fire and shock is covered.",
    "terrorism_exclusion": "This policy excludes loss or damage caused by terrorism or acts of political violence.",
    "war_exclusion": "Loss caused by war, invasion, civil war or rebellion is excluded from coverage.",
    "wear_tear_exclusion": "Damage due to wear and tear, gradual deterioration or rust is not covered.",
    "nuclear_exclusion": "Nuclear perils and radioactive contamination are excluded from this policy.",
    "premium_clause": "The insured shall pay the premium as specified in the schedule.",
    "notice_clause": "The insured shall give immediate notice to the company in case of any loss.",
    "arbitration_clause": "Disputes shall be resolved through arbitration as per applicable law.",
    "reinstatement_clause": "The company may at its option reinstate or replace damaged property.",
    "sum_insured_clause": "The maximum liability of the insurer shall not exceed the sum insured."
}

MODEL_NAME = "all-MiniLM-L6-v2"

class ClauseSimilarityChecker:
    def __init__(self):
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.standard_embeddings = self._embed_standards()
        print("Similarity model ready.")

    def _embed_standards(self):
        embeddings = {}
        for key, clause in STANDARD_CLAUSES.items():
            embeddings[key] = self.model.encode(clause, convert_to_numpy=True)
        return embeddings

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def check_document(self, sentences, threshold=0.45):
        if not sentences:
            return {}

        doc_embeddings = self.model.encode(sentences, convert_to_numpy=True)

        results = {}
        for clause_key, std_embedding in self.standard_embeddings.items():
            similarities = [
                self.cosine_similarity(std_embedding, sent_emb)
                for sent_emb in doc_embeddings
            ]
            best_score = max(similarities)
            best_sentence = sentences[np.argmax(similarities)]

            results[clause_key] = {
                "score": round(best_score, 4),
                "found": best_score >= threshold,
                "best_match": best_sentence if best_score >= threshold else None,
                "status": "PRESENT" if best_score >= threshold else "MISSING"
            }

        return results

    def get_anomalies(self, results):
        critical = ["fire_coverage", "terrorism_exclusion",
                    "war_exclusion", "premium_clause", "sum_insured_clause"]
        anomalies = []
        for key, val in results.items():
            if val["status"] == "MISSING":
                severity = "HIGH" if key in critical else "MEDIUM"
                anomalies.append({
                    "clause": key.replace("_", " ").title(),
                    "severity": severity,
                    "score": val["score"]
                })
        anomalies.sort(key=lambda x: (x["severity"] == "MEDIUM", x["score"]))
        return anomalies

    def get_summary_stats(self, results):
        total = len(results)
        present = sum(1 for v in results.values() if v["found"])
        missing = total - present
        avg_score = round(np.mean([v["score"] for v in results.values()]), 4)
        return {
            "total_clauses_checked": total,
            "present": present,
            "missing": missing,
            "coverage_percentage": round((present / total) * 100, 1),
            "avg_similarity_score": avg_score
        }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utils.pdf_parser import parse_pdf
    import os

    checker = ClauseSimilarityChecker()

    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in data/")
    else:
        test_pdf = os.path.join("data", pdf_files[0])
        print(f"\nTesting on: {pdf_files[0]}")
        doc = parse_pdf(test_pdf)
        sentences = doc["sentences"]
        print(f"Sentences: {len(sentences)}")

        results = checker.check_document(sentences)
        stats = checker.get_summary_stats(results)
        anomalies = checker.get_anomalies(results)

        print(f"\n── Clause Coverage ──")
        for key, val in results.items():
            icon = "✓" if val["found"] else "✗"
            print(f"  {icon} {key.replace('_',' ').title():<30} score={val['score']:.4f}  {val['status']}")

        print(f"\n── Summary ──")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print(f"\n── Anomalies ({len(anomalies)} found) ──")
        for a in anomalies:
            print(f"  [{a['severity']}] {a['clause']} (score={a['score']})")