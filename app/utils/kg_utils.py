from typing import List, Dict, Any
from neo4j import Driver
import math

def format_kg_path(path: List[Dict[str, Any]]) -> str:
    """
    Converts a path list into a readable string for LLM context injection.
    Handles missing keys gracefully to prevent runtime crashes.
    """
    if not path:
        return ""
    
    readable = []
    for item in path:
        start = item.get('start', 'Unknown')
        rel = item.get('relation', 'RELATED_TO')
        target = item.get('target', 'Unknown')
        desc = item.get('description')
        if desc:
            readable.append(f"{start} --[{rel}: {desc}]--> {target}")
        else:
            readable.append(f"{start} --[{rel}]--> {target}")
    
    return " | ".join(readable)

def calculate_graph_centrality(node_name: str, driver: Driver) -> float:
    """
    Calculates the Degree Centrality of a node using live Graph data.
    
    Research Justification: 
    Degree Centrality (Freeman, 1978) measures the number of direct connections 
    a node has. Nodes with high degree centrality are 'hubs' of information 
    and should carry higher weight in confidence scoring.
    
    Logic:
    1. Query Neo4j for the count of relationships connected to the node.
    2. Normalize the count to a 0.0 - 1.0 score using a saturation function.
    """
    query = """
    MATCH (n:Entity {name: $name})-[r]-()
    RETURN count(r) as degree
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, name=node_name)
            record = result.single()
            
            if not record:
                return 0.0
            
            degree = record["degree"]
            if degree == 0:
                return 0.0
            
            score = degree / (degree + 5.0)
            return round(score, 4)

    except Exception as e:
        print(f"Error calculating centrality for node '{node_name}': {e}")
        return 0.0