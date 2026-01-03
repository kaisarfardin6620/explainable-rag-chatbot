import json
import numpy as np
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from app.services.rag_pipeline import run_rag_pipeline
from app.services.embedding_service import get_embeddings
from app.services.kg_store import close as close_neo4j


def calculate_f1(predicted: str, truth: str) -> float:
    """
    Standard NLP metric: Measures word overlap.
    """
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(truth.lower().split())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = pred_tokens.intersection(truth_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def calculate_semantic_similarity(predicted: str, truth: str) -> float:
    """
    Advanced metric: Uses embeddings to see if the meaning is the same,
    even if words are different.
    """
    if not predicted or not truth:
        return 0.0
    embeddings = get_embeddings([predicted, truth])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)


def run_evaluation(dataset_path: str, output_file: str = "evaluation_results.json"):
    print(f"📉 Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    total_f1 = 0.0
    total_sim = 0.0
    total_time = 0.0
    
    print(f"🚀 Starting Evaluation on {len(dataset)} questions...\n")
    
    for i, item in enumerate(dataset):
        question = item['question']
        ground_truth = item['answer']
        
        print(f"[{i+1}/{len(dataset)}] Q: {question[:50]}...")
        
        start_time = time.time()
        
        try:
            response = run_rag_pipeline(question, session_id=9999)
            predicted_answer = response['answer']
            confidence = response['confidence']
            support_ratio = response.get('explanation', {}).get('confidence_signals', {}).get('claim_support_ratio', 0)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            predicted_answer = "ERROR"
            confidence = 0.0
            support_ratio = 0.0
            
        duration = time.time() - start_time
        
        f1 = calculate_f1(predicted_answer, ground_truth)
        sim = calculate_semantic_similarity(predicted_answer, ground_truth)
        
        total_f1 += f1
        total_sim += sim
        total_time += duration
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "f1_score": round(f1, 4),
            "semantic_score": round(sim, 4),
            "system_confidence": confidence,
            "claim_support_ratio": support_ratio,
            "latency": round(duration, 3)
        })

    avg_f1 = total_f1 / len(dataset)
    avg_sim = total_sim / len(dataset)
    avg_latency = total_time / len(dataset)
    
    final_report = {
        "summary": {
            "total_questions": len(dataset),
            "average_f1_score": round(avg_f1, 4),
            "average_semantic_similarity": round(avg_sim, 4),
            "average_latency_seconds": round(avg_latency, 3)
        },
        "details": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2)
        
    print("\n" + "="*40)
    print("📊 EVALUATION COMPLETE")
    print(f"   Avg F1 Score: {avg_f1:.4f}")
    print(f"   Avg Semantic Sim: {avg_sim:.4f}")
    print(f"   Results saved to: {output_file}")
    print("="*40)

if __name__ == "__main__":
    import os
    if not os.path.exists("test_dataset.json"):
        print("❌ Please create 'test_dataset.json' first.")
    else:
        run_evaluation("test_dataset.json")
        close_neo4j()