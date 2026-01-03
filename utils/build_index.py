"""
Entity Linking Index Builder for Neo4j Knowledge Graph
=======================================================
This script builds a vector index for entity linking using the E5 multilingual model.

Model: intfloat/multilingual-e5-base
- SOTA retrieval performance for multilingual text
- Requires specific prefixes for asymmetric retrieval:
  * "passage: " - prepended to documents/entities being indexed
  * "query: " - prepended to search queries at inference time

This asymmetric prefixing is CRITICAL for E5 models to work correctly.
The model was trained with this convention to distinguish between
the "what I'm searching for" (query) and "what I'm searching through" (passage).
"""

import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ==============================================================================
# CONFIGURATION - Modify these values as needed
# ==============================================================================

# Neo4j Connection Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")

# Index Configuration
TARGET_PROPERTY = "id"  # The node property to index (e.g., "ten", "name")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "entity_index.pkl")

# Model Configuration
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32

# E5 Model Prefixes (DO NOT CHANGE - required by the model)
PASSAGE_PREFIX = "passage: "  # For indexing entities
QUERY_PREFIX = "query: "      # For searching


# ==============================================================================
# DATA EXTRACTION
# ==============================================================================

def fetch_entity_names(
    uri: str,
    target_property: str
) -> List[str]:
    """
    Fetch all distinct entity names from Neo4j.
    
    This query is LABEL-AGNOSTIC - it sweeps through ALL nodes
    that have the target property, regardless of their label.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        target_property: The property key to index
        
    Returns:
        List of distinct entity names (strings)
    """
    driver = None
    names = []
    
    try:
        driver = GraphDatabase.driver(uri)
        
        # Verify connectivity
        driver.verify_connectivity()
        print(f"[SUCCESS] Connected to Neo4j at {uri}")
        
        with driver.session() as session:
            # Dynamic query - fetches ALL nodes with the target property
            # Using apoc-free syntax for maximum compatibility
            query = f"""
                MATCH (n)
                WHERE n.{target_property} IS NOT NULL 
                  AND n.{target_property} <> ''
                RETURN DISTINCT n.{target_property} AS name
            """
            
            result = session.run(query)
            records = list(result)
            
            # Extract names, filtering out any remaining nulls/empty strings
            names = [
                record["name"] 
                for record in records 
                if record["name"] and str(record["name"]).strip()
            ]
            
            print(f"[SUCCESS] Fetched {len(names)} distinct entity names")
            
    except Exception as e:
        print(f"[FAIL] Error connecting to Neo4j: {e}")
        raise
        
    finally:
        if driver:
            driver.close()
            print("[SUCCESS] Neo4j connection closed")
    
    return names


# ==============================================================================
# EMBEDDING GENERATION
# ==============================================================================

def generate_embeddings(
    names: List[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Generate embeddings for entity names using E5 model.
    
    CRITICAL E5 LOGIC:
    ------------------
    The E5 model family uses asymmetric prefixes for retrieval tasks:
    
    - "passage: " prefix is added to DOCUMENTS (the things being searched)
    - "query: " prefix is added to QUERIES (the search terms)
    
    This is because E5 was trained with contrastive learning where
    queries and passages occupy slightly different semantic spaces.
    Using the correct prefix ensures optimal retrieval performance.
    
    In this function, we add "passage: " because we're encoding
    the ENTITIES that will be searched through later.
    
    Args:
        names: List of entity names (raw, without prefix)
        model_name: HuggingFace model identifier
        batch_size: Batch size for encoding
        
    Returns:
        NumPy array of embeddings (shape: [n_entities, embedding_dim])
    """
    print(f"\n[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[SUCCESS] Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})")
    
    # MANDATORY: Add "passage: " prefix for E5 document encoding
    # This tells the model these are the "passages" to be retrieved
    prefixed_names = [f"{PASSAGE_PREFIX}{name}" for name in names]
    
    print(f"\n[INFO] Generating embeddings for {len(names)} entities...")
    print(f"   Example: '{names[0]}' -> '{prefixed_names[0]}'")
    
    embeddings = model.encode(
        prefixed_names,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    print(f"[SUCCESS] Embeddings generated: shape {embeddings.shape}")
    
    return embeddings


# ==============================================================================
# SERIALIZATION
# ==============================================================================

def save_index(
    names: List[str],
    embeddings: np.ndarray,
    output_path: str = OUTPUT_FILE
) -> None:
    """
    Save the entity index to a pickle file.
    
    Structure:
    {
        'names': [list of raw entity names - NO prefix],
        'embeddings': numpy array of shape (n_entities, embedding_dim)
    }
    
    NOTE: We save raw names (without "passage: " prefix) because:
    1. The prefix is only needed during encoding
    2. We want to display clean names to users
    3. Saves storage space
    """
    index_data = {
        'names': names,
        'embeddings': embeddings
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Index saved to '{output_path}' ({file_size_mb:.2f} MB)")


def load_index(index_path: str = OUTPUT_FILE) -> Dict[str, Any]:
    """Load the entity index from pickle file."""
    with open(index_path, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# SEARCH FUNCTION
# ==============================================================================

def search(
    query: str,
    index_data: Dict[str, Any],
    model: SentenceTransformer,
    top_k: int = 3
) -> List[tuple]:
    """
    Search for the most similar entities to a query.
    
    CRITICAL E5 LOGIC:
    ------------------
    When searching, we must use "query: " prefix (NOT "passage: ").
    
    The E5 model maps queries and passages to the same vector space,
    but with the understanding that:
    - Queries are typically short and express information needs
    - Passages are longer and contain the information
    
    Using "query: " tells the model to encode the input as a search query,
    which will be compared against passages (entities) encoded with "passage: ".
    
    Args:
        query: The search query (raw, without prefix)
        index_data: Dictionary with 'names' and 'embeddings'
        model: Loaded SentenceTransformer model
        top_k: Number of results to return
        
    Returns:
        List of (name, score) tuples, sorted by similarity
    """
    # MANDATORY: Add "query: " prefix for E5 query encoding
    prefixed_query = f"{QUERY_PREFIX}{query}"
    
    # Encode the query
    query_embedding = model.encode(
        prefixed_query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, index_data['embeddings'])[0]
    
    # Get top-k results
    top_indices = similarities.argsort(descending=True)[:top_k]
    
    results = [
        (index_data['names'][idx], float(similarities[idx]))
        for idx in top_indices
    ]
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def build_index() -> None:
    """Main function to build the entity linking index."""
    print("=" * 60)
    print("Entity Linking Index Builder")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Target Property: {TARGET_PROPERTY}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)
    
    # Step 1: Fetch entity names from Neo4j
    print("\n[1/3] Fetching entity names from Neo4j...")
    names = fetch_entity_names(
        NEO4J_URI, TARGET_PROPERTY
    )
    
    if not names:
        print("[FAIL] No entities found! Check your Neo4j connection and TARGET_PROPERTY.")
        return
    
    # Step 2: Generate embeddings
    print("\n[2/3] Generating embeddings...")
    embeddings = generate_embeddings(names)
    
    # Step 3: Save the index
    print("\n[3/3] Saving index...")
    save_index(names, embeddings)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Index building complete!")
    print("=" * 60)


# Need os for file size calculation
import os


if __name__ == "__main__":
    # Check for existing index file and warn
    if os.path.exists(OUTPUT_FILE):
        print(f"[WARNING] Output file '{OUTPUT_FILE}' already exists and will be overwritten.")
    # Build the index
    build_index()
    
    # ===========================================================================
    # SANITY CHECK / DEMO
    # ===========================================================================
    print("\n" + "=" * 60)
    print("[INFO] SEARCH DEMO (Sanity Check)")
    print("=" * 60)
    
    # Load the saved index
    print("\nLoading index...")
    index_data = load_index()
    print(f"[SUCCESS] Loaded {len(index_data['names'])} entities")
    
    # Load model for search
    print(f"\nLoading model for search...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Test queries (Vietnamese pharmaceutical terms)
    test_queries = [
        "long đởm",      # Gentiana (medicinal herb)
        "cam thảo",      # Licorice
        "đau đầu",       # Headache
        "ho",            # Cough
    ]
    
    print("\n" + "-" * 60)
    for query in test_queries:
        results = search(query, index_data, model, top_k=3)
        
        print(f"\n[INFO] Query: \"{query}\"")
        print(f"   (Encoded as: \"{QUERY_PREFIX}{query}\")")
        print("   Results:")
        for rank, (name, score) in enumerate(results, 1):
            print(f"     {rank}. {name} (score: {score:.4f})")
