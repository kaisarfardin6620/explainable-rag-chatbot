from openai import OpenAI
from app.core.config import settings
from typing import List, Dict, Any, Optional

client = OpenAI(api_key=settings.openai_api_key)

def generate_answer(
    question: str,
    evidence: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> str:
    """
    Generate answer using hybrid evidence (RAG chunks + KG paths).
    Strict grounding + refusal instructions.
    """
    context_parts = []

    if evidence.get("rag_evidence"):
        context_parts.append("=== Relevant Document Excerpts ===\n")
        for ev in evidence["rag_evidence"]:
            context_parts.append(
                f"Document: {ev['document']} (Page {ev['page']})\n"
                f"Text: {ev['text'][:1000]}\n"
            )

    if evidence.get("kg_evidence"):
        context_parts.append("=== Knowledge Graph Facts ===\n")
        for path in evidence["kg_evidence"][:10]:
            if "path" in path:
                rels = " -> ".join([f"{r['type']} ({r.get('description', '')})" for r in path["path"]])
                context_parts.append(f"{path.get('start', '')} {rels} {path.get('target', '')}\n")
            else:
                context_parts.append(str(path) + "\n")

    context = "\n".join(context_parts)

    system_prompt = """
You are a precise, evidence-based reasoning assistant.
Answer the user's question using ONLY the provided evidence below.
- If the evidence fully supports an answer, respond clearly and cite sources.
- If the evidence is insufficient or contradictory, say: 
  "I don't have sufficient evidence to answer this confidently."
- Do NOT hallucinate, speculate, or use external knowledge.
- Keep answers concise and factual.
    """.strip()

    user_prompt = f"""
Evidence:
{context}

Question: {question}

Answer:
    """.strip()

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0
    )

    return response.choices[0].message.content.strip()


def generate_structured(
    prompt: str,
    response_format: Optional[Dict] = None,
    temperature: float = 0.0
) -> str:
    """
    For structured tasks (e.g., entity extraction, JSON output).
    Use response_format={"type": "json_object"} when needed.
    """
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format=response_format
    )
    return response.choices[0].message.content.strip()


def extract_entities_relations(text: str) -> Dict[str, Any]:
    """
    Structured extraction for KG building (used in kg_builder.py)
    """
    prompt = f"""
You are an expert knowledge graph extractor.
Extract entities and relationships from the text below.

Return ONLY valid JSON in this format:
{{
  "entities": [{{"name": "string", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|DATE|OTHER"}}],
  "relationships": [
    {{
      "source": "exact entity name",
      "target": "exact entity name",
      "relation": "UPPERCASE_RELATION (e.g. WORKS_AT, LOCATED_IN)",
      "description": "short explanation"
    }}
  ]
}}

If nothing clear, return empty lists.

Text: {text}
    """

    import json
    raw = generate_structured(
        prompt=prompt,
        response_format={"type": "json_object"},
        temperature=0.0
    )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}