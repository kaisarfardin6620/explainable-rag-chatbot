import json
from typing import List, Dict, Any
from neo4j import GraphDatabase
from app.core.config import settings
from app.services.llm_service import client as openai_client 

driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_username, settings.neo4j_password),
    database=settings.neo4j_database
)

EXTRACTION_PROMPT = """
You are an expert knowledge graph builder. Extract entities and relationships from the given text.

Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "Entity Name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|DATE|OTHER"}
  ],
  "relationships": [
    {
      "source": "Exact entity name",
      "target": "Exact entity name",
      "relation": "Brief relation in uppercase (e.g. WORKS_AT, LOCATED_IN, ACQUIRED_BY, CAUSED)",
      "description": "One short sentence explaining the relation"
    }
  ]
}

Rules:
- Extract only clear, factual relations
- Do not hallucinate
- Use consistent entity names
- If no clear entities/relations, return empty lists

Text:
{text}
"""

def extract_entities_relations(text: str) -> Dict[str, Any]:
    """
    Use OpenAI to extract structured entities and relations from chunk text.
    """
    response = openai_client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "You are a precise knowledge extraction system."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1000
    )

    try:
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"KG extraction failed: {e}")
        return {"entities": [], "relationships": []}


def build_kg_from_chunks(
    chunks: List[Dict[str, Any]],
    document_name: str,
    document_id: int
) -> None:
    """
    Main function: build KG from list of chunks.
    Called after chunks are saved to SQLite and upserted to Pinecone.
    """
    with driver.session() as session:
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_id = chunk.get("chunk_id")
            page = chunk["metadata"].get("page", 0)

            extracted = extract_entities_relations(chunk_text)

            for entity in extracted.get("entities", []):
                session.run("""
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET e.type = $type, e.first_seen = timestamp()
                    ON MATCH SET e.last_seen = timestamp()
                    """, name=entity["name"], type=entity["type"])

            for rel in extracted.get("relationships", []):
                session.run("""
                    MATCH (source:Entity {name: $source})
                    MATCH (target:Entity {name: $target})
                    MERGE (source)-[r:RELATION {type: $relation}]->(target)
                    ON CREATE SET 
                        r.description = $description,
                        r.confidence = 0.9,
                        r.sources = [$source_info]
                    ON MATCH SET 
                        r.confidence = r.confidence + 0.1,
                        r.sources = r.sources + $source_info
                    """,
                    source=rel["source"],
                    target=rel["target"],
                    relation=rel["relation"],
                    description=rel["description"],
                    source_info={
                        "document": document_name,
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "page": page
                    }
                )

            entity_names = [e["name"] for e in extracted.get("entities", [])]
            if entity_names and chunk_id:
                session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $entity_names
                    """,
                    chunk_id=chunk_id,
                    entity_names=entity_names
                )

                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    ON CREATE SET 
                        c.text = $text,
                        c.document = $document,
                        c.page = $page
                    """,
                    chunk_id=chunk_id,
                    text=chunk_text[:1000],
                    document=document_name,
                    page=page
                )

def clear_kg() -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def close_driver():
    driver.close()