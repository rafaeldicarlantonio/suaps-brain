# memory/graph.py

import os
from typing import List, Dict, Any
from collections import defaultdict

# Edge-type weights (you can tune these per PRD / org priorities)
EDGE_WEIGHTS = {
    "decision": 1.0,
    "deadline": 0.9,
    "procedure": 0.8,
    "mentioned_in": 0.5,
}

def expand_entities(
    sb,
    base_memories: List[Dict[str, Any]],
    max_hops: int = 3,
    max_neighbors: int = 10,
    max_per_entity: int = 3,
) -> List[Dict[str, Any]]:
    """
    Expand retrieval results by traversing entity graph relationships.

    Args:
        sb: Supabase client
        base_memories: list of memory dicts (with "id", "type", etc.)
        max_hops: number of hops to traverse (default 3)
        max_neighbors: cap on total neighbor chunks to return
        max_per_entity: cap on neighbors per entity

    Returns:
        List of neighbor memory dicts with reason="graph_neighbor"
    """

    if not base_memories:
        return []

    neighbor_chunks: List[Dict[str, Any]] = []
    visited_entities = set()
    visited_memories = {m["id"] for m in base_memories}

    # Step 1: find all entities linked to base memories
    mem_ids = [m["id"] for m in base_memories]
    ent_rows = (
        sb.table("entity_mentions")
        .select("entity_id,memory_id")
        .in_("memory_id", mem_ids)
        .execute()
    )
    ent_data = ent_rows.data if hasattr(ent_rows, "data") else ent_rows.get("data") or []
    start_entities = {row["entity_id"] for row in ent_data}
    frontier = list(start_entities)
    visited_entities.update(start_entities)

    # Traverse graph up to max_hops
    for hop in range(1, max_hops + 1):
        if not frontier:
            break

        # Fetch neighbors via edges
        edge_rows = (
            sb.table("entity_edges")
            .select("src,dst,rel,weight")
            .in_("src", frontier)
            .execute()
        )
        edges = edge_rows.data if hasattr(edge_rows, "data") else edge_rows.get("data") or []

        new_entities = set()
        for e in edges:
            dst = e["dst"]
            rel = e.get("rel") or "related"
            base_weight = EDGE_WEIGHTS.get(rel, 0.5)
            if dst not in visited_entities:
                new_entities.add(dst)
            visited_entities.add(dst)

            # Fetch memories mentioning this neighbor entity
            mem_rows = (
                sb.table("entity_mentions")
                .select("memory_id,entity_id")
                .eq("entity_id", dst)
                .limit(max_per_entity)
                .execute()
            )
            mem_data = mem_rows.data if hasattr(mem_rows, "data") else mem_rows.get("data") or []

            mem_ids = [row["memory_id"] for row in mem_data if row["memory_id"] not in visited_memories]
            if not mem_ids:
                continue

            mrows = (
                sb.table("memories")
                .select("id,type,title,value,tags,created_at")
                .in_("id", mem_ids)
                .limit(max_per_entity)
                .execute()
            )
            mdata = mrows.data if hasattr(mrows, "data") else mrows.get("data") or []

            for r in mdata:
                visited_memories.add(r["id"])
                neighbor_chunks.append({
                    "id": r["id"],
                    "type": r.get("type") or "semantic",
                    "title": r.get("title") or "",
                    "text": f"[GRAPH NEIGHBOR via {rel.upper()} HOP {hop}] {r.get('value') or ''}",
                    "tags": r.get("tags") or [],
                    "created_at": r.get("created_at"),
                    "score": base_weight * (0.9 ** (hop - 1)),  # decay with hop depth
                    "reason": "graph_neighbor",
                })

        # Prepare next frontier
        frontier = list(new_entities)

        # Stop if we hit neighbor cap
        if len(neighbor_chunks) >= max_neighbors:
            break

    return neighbor_chunks[:max_neighbors]
