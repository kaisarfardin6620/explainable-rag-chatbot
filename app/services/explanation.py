from typing import Dict, Any, List
from app.services.verification import verify_claims

def build_explanation(
    answer: str,
    verification: Dict[str, Any],
    evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a structured, traceable explanation with:
    - Answer breakdown
    - Supporting evidence paths (RAG text + KG paths)
    - Unsupported claims (if any)
    - Source provenance
    """
    explanation = {
        "summary": "The answer is grounded in retrieved document chunks and knowledge graph paths.",
        "supported_claims": [],
        "unsupported_claims": verification.get("unsupported", []),
        "sources": {
            "documents": [],
            "kg_paths": []
        },
        "confidence_signals": {
            "retrieval_similarity": _avg_similarity(evidence.get("rag_evidence", [])),
            "kg_coverage": len(evidence.get("kg_evidence", [])),
            "claim_support_ratio": round(verification.get("support_ratio", 1.0), 3)
        }
    }

    seen_docs = set()
    for ev in evidence.get("rag_evidence", []):
        key = (ev["document"], ev["page"])
        if key not in seen_docs:
            explanation["sources"]["documents"].append({
                "document": ev["document"],
                "page": ev["page"],
                "type": "text_chunk"
            })
            seen_docs.add(key)

    for path in evidence.get("kg_evidence", [])[:5]:
        if "path" in path:
            path_str = " → ".join([
                f"{r['type']} ({r.get('description', '').strip()})" for r in path["path"]
            ])
            explanation["sources"]["kg_paths"].append({
                "path": f"{path.get('start', '?')} {path_str} {path.get('target', '?')}",
                "type": "kg_reasoning_path"
            })
        else:
            explanation["sources"]["kg_paths"].append({
                "fact": str(path),
                "type": "kg_fact"
            })

    claims = verification.get("claims", [])
    for claim in claims:
        if claim not in verification.get("unsupported", []):
            explanation["supported_claims"].append({
                "claim": claim,
                "supported_by": "text + KG" if _supported_by_both(claim, evidence) else 
                                "text" if _supported_by_text(claim, evidence) else "KG"
            })

    if verification["unsupported_count"] == 0:
        explanation["summary"] = "All claims in the answer are fully supported by retrieved evidence from documents and/or the knowledge graph."
    elif verification["unsupported_count"] < len(claims) / 2:
        explanation["summary"] = "Most claims are supported, but some lack direct evidence."
    else:
        explanation["summary"] = "Significant portions of the answer lack sufficient supporting evidence."

    return explanation


def _avg_similarity(rag_evidence: List[Dict]) -> float:
    if not rag_evidence:
        return 0.0
    return round(sum(ev["similarity"] for ev in rag_evidence) / len(rag_evidence), 3)


def _supported_by_text(claim: str, evidence: Dict) -> bool:
    claim_lower = claim.lower()
    for ev in evidence.get("rag_evidence", []):
        if claim_lower in ev["text"].lower():
            return True
    return False


def _supported_by_both(claim: str, evidence: Dict) -> bool:
    return _supported_by_text(claim, evidence) and any(
        claim.lower() in str(path).lower() for path in evidence.get("kg_evidence", [])
    )