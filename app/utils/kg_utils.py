from typing import List, Dict

def format_kg_path(path: List[Dict]) -> str:
    """Converts a path list into a readable string string."""
    if not path:
        return ""
    
    readable = []
    for item in path:
        readable.append(f"{item.get('start', '?')} --[{item.get('relation', '')}]--> {item.get('target', '?')}")
    
    return " | ".join(readable)

def calculate_graph_centrality(node_name: str, driver) -> float:
    """Placeholder for advanced centrality scoring."""
    return 0.5