from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from app.core.config import settings

driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_username, settings.neo4j_password),
    database=settings.neo4j_database
)

def search_entities(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fuzzy search for entities by name.
    Useful for query analysis and routing.
    """
    with driver.session() as session:
        result = session.run("""
            CALL db.index.fulltext.queryNodes("entityNameIndex", $query + "~")
            YIELD node, score
            RETURN node.name AS name, node.type AS type, score
            ORDER BY score DESC
            LIMIT $limit
            """, query=query, limit=limit)
        
        return [dict(record) for record in result]

def get_related_entities(entity_name: str, depth: int = 1) -> List[Dict[str, Any]]:
    """
    Get connected entities and relationships (multi-hop).
    Returns structured paths for reasoning.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH path = (e:Entity {name: $name})-[:RELATION*1..$depth]-(related)
            RETURN 
                e.name AS start,
                [r IN relationships(path) | {
                    type: r.type,
                    description: r.description,
                    confidence: coalesce(r.confidence, 0.9)
                }] AS relations,
                related.name AS target,
                related.type AS target_type
            ORDER BY length(path) DESC
            """, name=entity_name, depth=depth)
        
        paths = []
        for record in result:
            paths.append({
                "start": record["start"],
                "target": record["target"],
                "target_type": record["target_type"],
                "path": record["relations"]
            })
        return paths

def get_evidence_for_claim(
    claim_entities: List[str],
    max_paths: int = 5
) -> List[Dict[str, Any]]:
    """
    Find KG paths connecting entities mentioned in a claim/question.
    Used in hybrid retrieval and verification.
    """
    if len(claim_entities) < 2:
        return []

    with driver.session() as session:
        result = session.run("""
            UNWIND $entities AS entity1
            UNWIND $entities AS entity2
            WITH entity1, entity2 WHERE entity1 <> entity2
            MATCH path = shortestPath((e1:Entity {name: entity1})-[*..4]-(e2:Entity {name: entity2}))
            RETURN 
                entity1 AS source,
                entity2 AS target,
                [r IN relationships(path) | {
                    type: r.type,
                    description: r.description
                }] AS path,
                length(path) AS hops
            ORDER BY hops ASC
            LIMIT $max_paths
            """, entities=claim_entities, max_paths=max_paths)
        
        return [dict(record) for record in result]

def get_chunk_entities(chunk_id: str) -> List[str]:
    """
    Get all entities mentioned in a specific chunk (for hybrid scoring).
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
            RETURN e.name AS entity_name
            """, chunk_id=chunk_id)
        return [record["entity_name"] for record in result]

def get_provenance(relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get source provenance for relations (for citations).
    """
    with driver.session() as session:
        query = """
            MATCH ()-[r:RELATION]->()
            WHERE $rel_type IS NULL OR r.type = $rel_type
            UNWIND r.sources AS src
            RETURN 
                r.type AS relation,
                src.document AS document,
                src.page AS page,
                src.chunk_id AS chunk_id
            """
        result = session.run(query, rel_type=relation_type)
        return [dict(record) for record in result]

def create_indexes() -> None:
    """Create full-text index for fast entity search (run once)"""
    with driver.session() as session:
        session.run("""
            CREATE FULLTEXT INDEX entityNameIndex IF NOT EXISTS
            FOR (e:Entity) ON (e.name)
            OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}
            """)

create_indexes()

def close():
    driver.close()