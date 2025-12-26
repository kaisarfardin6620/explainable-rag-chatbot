from typing import List, Dict, Any, Tuple
from app.services.vector_store import query as pinecone_query
from app.services.kg_store import get_related_entities, get_evidence_for_claim
from app.services.llm_service import generate_with_evidence
from app.services.verification import verify_claims
from app.services.explanation import build_explanation
from app.core.config import settings
from app.database.repository import get_chunk_texts

def hybrid_retrieval(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Step 1: Hybrid retrieval – combine vector (Pinecone) + symbolic (Neo4j)
    """
    vector_results = pinecone_query(
        question=question,
        top_k=top_k,
        min_similarity=settings.min_similarity_threshold
    )

    chunk_ids = [match["chunk_id"] for match in vector_results]
    chunk_texts = get_chunk_texts(chunk_ids)

    rag_evidence = [
        {
            "text": chunk_texts.get(cid, ""),
            "document": match["document"],
            "page": match["page"],
            "similarity": match["similarity"],
            "source": "vector"
        }
        for cid, match in zip(chunk_ids, vector_results)
    ]

    entities = extract_entities_from_question(question)

    kg_evidence = []
    for entity in entities[:3]:
        paths = get_related_entities(entity, depth=2)
        kg_evidence.extend(paths)

    if len(entities) >= 2:
        claim_paths = get_evidence_for_claim(entities)
        kg_evidence.extend([{"source": "claim_path", **p} for p in claim_paths])

    return {
        "rag_evidence": rag_evidence,
        "kg_evidence": kg_evidence,
        "entities": entities
    }

def extract_entities_from_question(question: str) -> List[str]:
    """
    Lightweight entity extraction from question using LLM.
    """
    prompt = f"""
    Extract the main entities (people, organizations, locations, concepts) from this question.
    Return only a JSON list of strings.

    Question: {question}

    Entities:
    """
    response = generate_with_evidence(
        messages=[{"role": "user", "content": prompt}],
        evidence=[],
        force_json=True
    )
    try:
        import json
        return json.loads(response)
    except:
        return [word for word in question.split() if word.istitle()][:5]

def run_rag_pipeline(question: str, session_id: int) -> Dict[str, Any]:
    """
    Full hybrid KG-RAG pipeline for /chat/ask
    """
    evidence = hybrid_retrieval(question, top_k=settings.top_k)

    if not evidence["rag_evidence"] and not evidence["kg_evidence"]:
        return {
            "answer": "I don't have sufficient evidence to answer this question.",
            "refusal": True,
            "confidence": 0.0,
            "citations": [],
            "explanation": "No relevant chunks or KG paths found."
        }

    answer = generate_with_evidence(
        question=question,
        evidence=evidence
    )

    verification = verify_claims(answer, evidence)

    confidence = calculate_confidence(verification, evidence)

    if verification["unsupported_count"] > 0 or confidence < 0.5:
        return {
            "answer": "I cannot confidently answer this based on the available evidence.",
            "refusal": True,
            "confidence": confidence,
            "unsupported_claims": verification["unsupported"],
            "explanation": "Some claims lack supporting evidence from documents or knowledge graph."
        }

    explanation = build_explanation(answer, verification, evidence)

    return {
        "answer": answer,
        "confidence": round(confidence, 2),
        "confidence_level": "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low",
        "citations": extract_citations(evidence),
        "explanation": explanation,
        "refusal": False,
        "sources": {
            "vector_chunks": len(evidence["rag_evidence"]),
            "kg_paths": len(evidence["kg_evidence"])
        }
    }

def calculate_confidence(verification: Dict, evidence: Dict) -> float:
    """
    Calibrated confidence: combine retrieval + KG + verification signals
    """
    sim_score = sum(e["similarity"] for e in evidence["rag_evidence"]) / len(evidence["rag_evidence"] or [1])
    kg_coverage = len(evidence["kg_evidence"]) / 10  # Normalized
    claim_support = 1 - (verification["unsupported_count"] / max(len(verification["claims"]), 1))

    return (sim_score * 0.4) + (kg_coverage * 0.3) + (claim_support * 0.3)

def extract_citations(evidence: Dict) -> List[Dict]:
    citations = []
    seen = set()
    for ev in evidence["rag_evidence"]:
        key = (ev["document"], ev["page"])
        if key not in seen:
            citations.append({"document": ev["document"], "page": ev["page"], "source": "text"})
            seen.add(key)
    return citations