"""
MemeScope Evaluation Module
Computes ROUGE-L and BERTScore against human reference explanations.
"""

import json
from pathlib import Path


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)
    return {k: round(sum(v) / len(v), 4) for k, v in scores.items()}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore F1."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }


def evaluate_results(
    results_file: str,
    references_file: str,
    output_file: str = "outputs/evaluation.json",
) -> dict:
    """
    Evaluate pipeline outputs against human references.

    results_file: JSON from explain_meme_batch()
    references_file: JSON list of {"image": path, "reference": "human explanation text"}
    """
    with open(results_file) as f:
        results = json.load(f)
    with open(references_file) as f:
        references = json.load(f)

    ref_map = {r["image"]: r["reference"] for r in references}
    eval_output = {}

    for strategy in ["zero_shot", "few_shot"]:
        preds, refs = [], []
        for entry in results:
            img = entry["image"]
            if img not in ref_map:
                continue
            result = entry["results"].get(strategy, {})
            if "error" in result:
                continue
            full_pred = " ".join([
                result.get("visual", ""),
                result.get("text_meaning", ""),
                result.get("cultural_context", ""),
            ]).strip()
            preds.append(full_pred)
            refs.append(ref_map[img])

        if not preds:
            continue

        print(f"[Eval] Computing metrics for {strategy} ({len(preds)} examples)...")
        rouge = compute_rouge(preds, refs)
        bert = compute_bertscore(preds, refs)
        eval_output[strategy] = {"rouge": rouge, "bertscore": bert, "n_samples": len(preds)}
        print(f"  ROUGE-L: {rouge['rougeL']}  |  BERTScore F1: {bert['f1']}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"[Eval] Saved to {output_file}")
    return eval_output


def print_summary_table(eval_output: dict):
    """Print a formatted summary table."""
    print("\n" + "="*60)
    print(f"{'Method':<20} {'ROUGE-L':<12} {'BERTScore F1':<15} {'N'}")
    print("="*60)
    for method, metrics in eval_output.items():
        r = metrics["rouge"]["rougeL"]
        b = metrics["bertscore"]["f1"]
        n = metrics["n_samples"]
        print(f"{method:<20} {r:<12} {b:<15} {n}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Path to results JSON from pipeline")
    parser.add_argument("references", help="Path to references JSON")
    args = parser.parse_args()
    output = evaluate_results(args.results, args.references)
    print_summary_table(output)
