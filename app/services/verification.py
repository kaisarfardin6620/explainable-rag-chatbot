import json
from typing import List, Dict, Any
from app.services.llm_service import client, generate_structured
from app.core.config import settings

def verify_claims_nli(answer: str, evidence_texts: List[str]) -> Dict[str, Any]:
    """
    Research-Grade Verification:
    1. Extract atomic claims from the answer.
    2. Check if evidence ENTAILS the claim (not just keyword match).
    """
    
    claims = _extract_atomic_claims(answer)
    if not claims:
        return {"score": 0.0, "details": []}

    evidence_blob = "\n".join(evidence_texts)[:10000]
    
    verified_claims = []
    supported_count = 0

    for claim in claims:
        is_supported = _check_entailment(claim, evidence_blob)
        verified_claims.append({
            "claim": claim,
            "supported": is_supported
        })
        if is_supported:
            supported_count += 1
            
    score = supported_count / len(claims) if claims else 0.0
    
    return {
        "support_score": score,
        "claims": verified_claims
    }

def _extract_atomic_claims(text: str) -> List[str]:
    prompt = f"""
    Split the following text into atomic, factual claims. 
    Return JSON: {{"claims": ["claim1", "claim2"]}}
    Text: {text}
    """
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get("claims", [])
    except:
        return []

def _check_entailment(claim: str, evidence: str) -> bool:
    """
    Returns True if evidence supports claim.
    """
    prompt = f"""
    Evidence: {evidence}
    
    Claim: {claim}
    
    Does the evidence fully support this claim? Respond with JSON: {{"supported": true/false}}
    """
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get("supported", False)
    except:
        return False