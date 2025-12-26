import re
import json
from typing import List, Dict, Any
from app.services.llm_service import generate_structured

def extract_claims(answer: str) -> List[str]:
    """
    Extract atomic claims from the LLM answer using structured LLM call.
    Returns list of factual statements.
    """
    prompt = f"""
    You are a claim extraction expert.
    Break the following answer into individual atomic, verifiable claims.
    Return ONLY a JSON object with this format: {{ "claims": ["claim 1", "claim 2"] }}

    Rules:
    - One claim per fact
    - No reasoning, no questions
    - Include quantities, dates, entities, relations
    - If no clear claims, return empty list

    Answer:
    {answer}
    """

    raw = generate_structured(
        prompt=prompt,
        response_format={"type": "json_object"},
        temperature=0.0
    )

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "claims" in data:
            return [c.strip() for c in data["claims"] if c.strip()]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    sentences = re.split(r'(?<=[.!?])\s+', answer)
    return [s.strip() for s in sentences if s.strip() and len(s) > 10]


def claim_supported_by_text(claim: str, rag_evidence: List[Dict]) -> bool:
    """
    Check if claim is directly supported by any RAG chunk text.
    Simple string overlap + keyword match (conservative).
    """
    claim_lower = claim.lower()
    for ev in rag_evidence:
        text = ev["text"].lower()
        claim_words = set(claim_lower.split())
        text_words = set(text.split())
        if not claim_words:
            continue
        if len(claim_words & text_words) / len(claim_words) > 0.5:
            return True
    return False


def claim_supported_by_kg(claim: str, kg_evidence: List[Dict]) -> bool:
    """
    Check if claim is supported by any KG path.
    Looks for entity-relation matches.
    """
    claim_lower = claim.lower()
    for path in kg_evidence:
        if "path" in path:
            for rel in path["path"]:
                rel_desc = rel.get("description", "").lower()
                rel_type = rel.get("type", "").lower()
                full_rel = f"{rel_type} {rel_desc}".strip()
                if any(key in claim_lower for key in [rel_type, rel_desc, full_rel] if len(key) > 3):
                    return True
    return False


def verify_claims(answer: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main verification function.
    Returns detailed verification results for confidence & refusal.
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "claims": [],
            "unsupported": [],
            "unsupported_count": 0,
            "support_ratio": 1.0
        }

    rag_evidence = evidence.get("rag_evidence", [])
    kg_evidence = evidence.get("kg_evidence", [])

    unsupported = []
    rag_count = 0
    kg_count = 0

    for claim in claims:
        supported = False
        
        if claim_supported_by_text(claim, rag_evidence):
            supported = True
            rag_count += 1
        
        if not supported and claim_supported_by_kg(claim, kg_evidence):
            supported = True
            kg_count += 1

        if not supported:
            unsupported.append(claim)

    support_ratio = (len(claims) - len(unsupported)) / len(claims)

    return {
        "claims": claims,
        "unsupported": unsupported,
        "unsupported_count": len(unsupported),
        "support_ratio": support_ratio,
        "rag_support_count": rag_count,
        "kg_support_count": kg_count
    }