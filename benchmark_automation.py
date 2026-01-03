import json
import time
from app.services.rag_pipeline import run_rag_pipeline
from app.services.kg_store import close as close_neo4j

def calculate_f1(predicted: str, truth: str) -> float:
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(truth.lower().split())
    if not pred_tokens or not truth_tokens: return 0.0
    common = pred_tokens.intersection(truth_tokens)
    if not common: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

def run_ablation_study(dataset_file: str):
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    modes = ["llm_only", "rag_only", "kg_only", "hybrid"]
    
    print(f"🔬 STARTING ABLATION STUDY ON {len(dataset)} QUESTIONS")

    for mode in modes:
        print(f"\n👉 Running Mode: {mode.upper()}...")
        results = []
        total_f1 = 0
        
        for item in dataset:
            q = item['question']
            truth = item['answer']
            
            try:
                response = run_rag_pipeline(q, session_id=0, mode=mode)
                
                f1 = calculate_f1(response.get("answer", ""), truth)
                total_f1 += f1
                
                results.append({
                    "question": q,
                    "ground_truth": truth,
                    "predicted": response.get("answer", ""),
                    "f1_score": round(f1, 4),
                    "confidence": response.get("confidence", 0)
                })
            except Exception as e:
                print(f"Error on {q}: {e}")

        avg_f1 = total_f1 / len(dataset)
        output_filename = f"results_{mode}.json"
        
        final_data = {
            "mode": mode,
            "average_f1": round(avg_f1, 4),
            "details": results
        }
        
        with open(output_filename, "w") as f:
            json.dump(final_data, f, indent=2)
            
        print(f"✅ Finished {mode}. Avg F1: {avg_f1:.4f}. Saved to {output_filename}")

    close_neo4j()

if __name__ == "__main__":
    import os
    if not os.path.exists("test_dataset.json"):
        print("❌ Create 'test_dataset.json' first with 5-10 QA pairs!")
    else:
        run_ablation_study("test_dataset.json")