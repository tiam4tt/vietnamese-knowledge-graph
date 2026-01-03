"""
Vietnamese Knowledge Graph QA System
=====================================
A Streamlit application for querying a Vietnamese Pharmaceutical Knowledge Graph
using a fine-tuned BartPho model for NLQ-to-Cypher translation with Gemini refinement.

Pipeline:
1. Local BartPho model generates draft Cypher query
2. Gemini Flash refines syntax errors
3. Entity linking using E5 embeddings replaces entity names
4. Execute query on Neo4j
5. Gemini generates final Vietnamese answer
"""

import os
import re
import pickle
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google import genai
from google.genai import types

import dotenv
dotenv.load_dotenv()
# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")

# Model Paths
LOCAL_MODEL_PATH = "./models/bartpho-syllable-NLQ2Cypher"
HUGGINGFACE_MODEL_ID = "tiam4tt/bartpho-NLQ-2-cypher-ViPharma"
ENTITY_INDEX_PATH = "./data/entity_index.pkl"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"

# E5 Prefixes (Required by the model)
QUERY_PREFIX = "query: "
# Entity Linking Configuration
SIMILARITY_THRESHOLD = 0.85

# Generation Parameters
MAX_NEW_TOKENS = 128
NUM_BEAMS = 5
TEMPERATURE = 0.7
TOP_P = 0.9

# Database Schema for Gemini Context
DATABASE_SCHEMA = """
[Node Labels]
- DRUG: Ten che pham, vac xin, sinh pham.
- CHEMICAL: Hoa chat, hoat chat, ta duoc, thuoc thu.
- DISEASE: Benh ly hoac trieu chung lam sang.
- ORGANISM: Vi khuan, virus, dong te bao.
- TEST_METHOD: Ky thuat kiem nghiem.
- STANDARD: Chi so ky thuat (pH, nong do, hieu gia,...).
- STORAGE_CONDITION: Dieu kien moi truong luu giu.
- PRODUCTION_METHOD: Cong nghe bao che/san xuat.

[Relationship Types]
- TREATS: Thuoc/Hoat chat dieu tri Benh.
- CONTAINS: Thanh phan/Ta duoc co trong Thuoc.
- TARGETS: Hoat chat tac dong len Vi sinh vat/Co quan.
- HAS_STANDARD: Thuc the co Chi so ky thuat/Tieu chuan.
- TESTED_BY: Thuoc duoc kiem nghiem bang Phuong phap.
- REQUIRES: Phuong phap can co Hoa chat/Thuoc thu/Thiet bi.
- PRODUCED_BY: Thuoc duoc san xuat boi Phuong phap/Vi sinh vat.
- STORED_AT: Thuoc duoc bao quan tai Dieu kien moi truong.

[Node Properties]
- All nodes have an 'id' property which is the main identifier (e.g., "Paracetamol", "Dau dau")
- All nodes have a 'type' property matching the node label
"""

# Schema text for model input (matching training format)
SCHEMA_TEXT = """[N]
- DRUG: Ten che pham, vac xin, sinh pham.
- CHEMICAL: Hoa chat, hoat chat, ta duoc, thuoc thu.
- DISEASE: Benh ly hoac trieu chung lam sang.
- ORGANISM: Vi khuan, virus, dong te bao.
- TEST_METHOD: Ky thuat kiem nghiem.
- STANDARD: Chi so ky thuat (pH, nong do, hieu gia,...).
- STORAGE_CONDITION: Dieu kien moi truong luu giu.
- PRODUCTION_METHOD: Cong nghe bao che/san xuat.

[R]
- TREATS: Thuoc/Hoat chat dieu tri Benh.
- CONTAINS: Thanh phan/Ta duoc co trong Thuoc.
- TARGETS: Hoat chat tac dong len Vi sinh vat/Co quan.
- HAS_STANDARD: Thuc the co Chi so ky thuat/Tieu chuan.
- TESTED_BY: Thuoc duoc kiem nghiem bang Phuong phap.
- REQUIRES: Phuong phap can co Hoa chat/Thuoc thu/Thiet bi.
- PRODUCED_BY: Thuoc duoc san xuat boi Phuong phap/Vi sinh vat.
- STORED_AT: Thuoc duoc bao quan tai Dieu kien moi truong.
"""


# ==============================================================================
# RESOURCE LOADING (Cached)
# ==============================================================================

@st.cache_resource
def load_neo4j_driver():
    """Initialize Neo4j driver connection."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI)
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None


@st.cache_resource
def load_cypher_model():
    """Load the fine-tuned BartPho NLQ-to-Cypher model.
    
    First attempts to load from local path. If not found, downloads from HuggingFace.
    """
    try:
        # Check if local model exists
        if os.path.exists(LOCAL_MODEL_PATH):
            st.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
            model_path = LOCAL_MODEL_PATH
        else:
            st.warning(f"Local model not found at {LOCAL_MODEL_PATH}")
            st.info(f"Downloading model from HuggingFace: {HUGGINGFACE_MODEL_ID}")
            model_path = HUGGINGFACE_MODEL_ID
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        model.eval()
        st.success(f"Model loaded successfully from {'local path' if model_path == LOCAL_MODEL_PATH else 'HuggingFace'}")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load Cypher model: {e}")
        return None, None


@st.cache_resource
def load_e5_model():
    """Load the E5 multilingual embedding model."""
    try:
        model = SentenceTransformer(E5_MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Failed to load E5 model: {e}")
        return None


@st.cache_resource
def load_entity_index():
    """Load the pre-computed entity index."""
    try:
        with open(ENTITY_INDEX_PATH, 'rb') as f:
            index_data = pickle.load(f)
        return index_data
    except Exception as e:
        st.error(f"Failed to load entity index: {e}")
        return None


@st.cache_resource
def get_gemini_client(api_key: str) -> Optional[genai.Client]:
    """Create and cache a Gemini client with API key."""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return None


# ==============================================================================
# STEP 1: LOCAL CYPHER GENERATION
# ==============================================================================

def generate_draft_cypher(
    question: str,
    tokenizer,
    model,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_beams: int = NUM_BEAMS
) -> str:
    """
    Generate a draft Cypher query using the local BartPho model.
    
    Args:
        question: Natural language question in Vietnamese
        tokenizer: BartPho tokenizer
        model: Fine-tuned BartPho model
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search
        
    Returns:
        Draft Cypher query string
    """
    # Format input to match training format
    input_text = f"[Q]\n{question}\n\n{SCHEMA_TEXT}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            early_stopping=True
        )
    
    # Decode
    draft_cypher = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return draft_cypher.strip()


# ==============================================================================
# GEMINI CONTENT GENERATION UTILITIES
# ==============================================================================

def gemini_generate_content(
    gemini_client: genai.Client,
    model_name: str,
    prompt: str,
    thinking_budget: int = 0
) -> str:
    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            )
        )
        return response.text.strip()
    except Exception as e:
        raise e


# ==============================================================================
# STEP 2: GEMINI SYNTAX REFINEMENT
# ==============================================================================

def refine_cypher_syntax(
    draft_cypher: str, 
    gemini_client: genai.Client,
    model_name: str = "models/gemini-2.5-flash"
) -> str:
    """
    Use Gemini to fix syntax errors in the Cypher query.
    
    Args:
        draft_cypher: The draft Cypher query from local model
        gemini_client: Gemini client instance
        model_name: Gemini model to use
        
    Returns:
        Syntax-refined Cypher query
    """
    prompt = f"""You are a Neo4j Cypher expert. Fix any syntax errors in this Cypher query for Neo4j.
Do NOT change the entity names inside quotes yet. Only fix structural syntax issues.
Ensure the syntax is valid Neo4j Cypher.

Database Schema:
{DATABASE_SCHEMA}

Draft Query:
{draft_cypher}

Return ONLY the corrected Cypher query string. No explanations, no markdown, no code blocks.
If the query is already correct, return it as-is."""

    try:
        refined = gemini_generate_content(gemini_client=gemini_client, model_name=model_name, prompt=prompt)
        
        # Clean up any accidental markdown formatting
        refined = re.sub(r'^```(?:cypher)?\s*', '', refined)
        refined = re.sub(r'\s*```$', '', refined)
        
        return refined.strip()
    except Exception as e:
        st.warning(f"Gemini refinement failed: {e}. Using draft query.")
        return draft_cypher


# ==============================================================================
# STEP 3: ENTITY LINKING & REPLACEMENT
# ==============================================================================

def extract_quoted_strings(cypher_query: str) -> List[str]:
    """Extract all strings inside double quotes from the Cypher query."""
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, cypher_query)
    return matches


def find_best_entity_match(
    term: str,
    e5_model: SentenceTransformer,
    entity_index: Dict[str, Any],
    threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[Optional[str], float]:
    """
    Find the best matching entity for a term using E5 embeddings.
    
    Args:
        term: The term to match
        e5_model: E5 embedding model
        entity_index: Pre-computed entity index
        threshold: Similarity threshold
        
    Returns:
        Tuple of (best_match_name or None, similarity_score)
    """
    # Add query prefix for E5 model
    prefixed_query = f"{QUERY_PREFIX}{term}"
    
    # Encode the query
    query_embedding = e5_model.encode(
        prefixed_query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, entity_index['embeddings'])[0]
    
    # Get best match
    best_idx = int(similarities.argmax().item())
    best_score = float(similarities[best_idx].item())
    
    if best_score > threshold:
        return entity_index['names'][best_idx], best_score
    else:
        return None, best_score


def replace_entities_in_query(
    cypher_query: str,
    e5_model: SentenceTransformer,
    entity_index: Dict[str, Any],
    threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Replace entity names in the Cypher query with matched entities from the index.
    
    Args:
        cypher_query: The Cypher query with potential entity names
        e5_model: E5 embedding model
        entity_index: Pre-computed entity index
        threshold: Similarity threshold for replacement
        
    Returns:
        Tuple of (modified_query, list of replacement details)
    """
    # Extract quoted strings
    quoted_terms = extract_quoted_strings(cypher_query)
    
    replacements = []
    modified_query = cypher_query
    
    for term in quoted_terms:
        if not term.strip():
            continue
            
        best_match, score = find_best_entity_match(
            term, e5_model, entity_index, threshold
        )
        
        replacement_info = {
            'original': term,
            'matched': best_match if best_match else term,
            'score': score,
            'replaced': best_match is not None
        }
        replacements.append(replacement_info)
        
        # Only replace if we found a match above threshold
        if best_match and best_match != term:
            # Replace the term in the query (keeping the quotes)
            modified_query = modified_query.replace(f'"{term}"', f'"{best_match}"')
    
    return modified_query, replacements


# ==============================================================================
# STEP 4: QUERY EXECUTION
# ==============================================================================

def execute_cypher_query(
    driver,
    cypher_query: str
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Execute a Cypher query against Neo4j.
    
    Args:
        driver: Neo4j driver instance
        cypher_query: The Cypher query to execute
        
    Returns:
        Tuple of (results list, error message or None)
    """
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [dict(record) for record in result]
            return records, None
    except Exception as e:
        return [], str(e)


# ==============================================================================
# STEP 5: FINAL ANSWER GENERATION
# ==============================================================================

def generate_final_answer(
    question: str,
    cypher_query: str,
    execution_results: List[Dict[str, Any]],
    error: Optional[str],
    gemini_client: genai.Client,
    model_name: str = "models/gemini-2.5-flash"
) -> str:
    """
    Generate a natural language answer in Vietnamese using Gemini.
    
    Args:
        question: Original user question
        cypher_query: The executed Cypher query
        execution_results: Results from Neo4j
        error: Any execution error
        gemini_client: Gemini client instance
        model_name: Gemini model to use
        
    Returns:
        Natural language answer in Vietnamese
    """
    if error:
        results_text = f"Query execution error: {error}"
    elif not execution_results:
        results_text = "No results found in the database."
    else:
        # Format results
        results_text = "Database Results:\n"
        for i, record in enumerate(execution_results[:20], 1):  # Limit to 20 results
            results_text += f"{i}. {record}\n"
        
        if len(execution_results) > 20:
            results_text += f"... and {len(execution_results) - 20} more results."
    
    prompt = f"""You are a helpful assistant for a Vietnamese Pharmaceutical Knowledge Graph system.
Based on the database query results, answer the user's question naturally in Vietnamese.

User Question: {question}

Cypher Query Executed:
{cypher_query}

{results_text}

Instructions:
- Answer in Vietnamese
- Be concise and informative
- If no results were found, politely explain that no information was found in the database
- If there was an error, explain that there was a technical issue with the query
- Do not include technical details like Cypher syntax in your answer
- Present the information in a clear, user-friendly manner"""

    try:
        response = gemini_generate_content(gemini_client=gemini_client, model_name=model_name, prompt=prompt)
        return response
    except Exception as e:
        if execution_results:
            # Fallback: format results directly
            result_strs = []
            for record in execution_results[:10]:
                for key, value in record.items():
                    result_strs.append(str(value))
            return "Result: " + ", ".join(result_strs)
        elif error:
            return f"An error occured while executing query: {error}"
        else:
            return "Couldn't find your answer in the database."


# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Vietnamese Knowledge Graph QA",
        page_icon=None,
        layout="wide"
    )
    
    st.title("Vietnamese Pharmaceutical Knowledge Graph QA System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input (load from env or user input)
        api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Similarity threshold
        sim_threshold = st.slider(
            "Entity Matching Threshold",
            min_value=0.5,
            max_value=1.0,
            value=SIMILARITY_THRESHOLD,
            step=0.05,
            help="Minimum similarity score for entity replacement"
        )
        
        # Gemini model selection
        gemini_model = st.selectbox(
            "Gemini Model",
            options=["models/gemini-2.5-flash", "models/gemini-2.5-pro"],
            index=0,
            help="Select the Gemini model for refinement"
        )
        
        st.markdown("---")
        st.markdown("**System Status**")
    
    # Initialize resources
    with st.sidebar:
        # Neo4j connection
        driver = load_neo4j_driver()
        if driver:
            st.success("Neo4j: Connected")
        else:
            st.error("Neo4j: Disconnected")
        
        # Cypher model
        tokenizer, cypher_model = load_cypher_model()
        if tokenizer and cypher_model:
            st.success("Cypher Model: Loaded")
        else:
            st.error("Cypher Model: Failed")
        
        # E5 model
        e5_model = load_e5_model()
        if e5_model:
            st.success("E5 Model: Loaded")
        else:
            st.error("E5 Model: Failed")
        
        # Entity index
        entity_index = load_entity_index()
        if entity_index:
            st.success(f"Entity Index: {len(entity_index['names'])} entities")
        else:
            st.error("Entity Index: Failed")
        
        # Gemini configuration
        gemini_client = None
        if api_key:
            gemini_client = get_gemini_client(api_key)
            if gemini_client:
                st.success("Gemini: Configured")
            else:
                st.error("Gemini: Failed")
        else:
            st.warning("Gemini: API key required")
    
    # Main input area
    st.subheader("Enter Your Question")
    question = st.text_area(
        "Question",
        placeholder="VD: Những bệnh nào có thể điều trị bằng thuốc Rifampicin?",
        height=100,
        label_visibility="collapsed"
    )
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        process_btn = st.button("Get Response", type="primary", use_container_width=True)
    
    # Check if all resources are available
    resources_ready = all([
        driver is not None,
        tokenizer is not None,
        cypher_model is not None,
        e5_model is not None,
        entity_index is not None,
        gemini_client is not None
    ])
    
    if process_btn:
        if not question.strip():
            st.warning("Please enter a question.")
            return
        
        if not resources_ready:
            st.error("System resources not fully initialized. Please check the sidebar for status.")
            return
        
        # Processing pipeline
        with st.spinner("Processing your question..."):
            
            # Step 1: Generate draft Cypher
            st.info("Step 1/4: Generating draft Cypher query...")
            draft_cypher = generate_draft_cypher(
                question, tokenizer, cypher_model
            )
            
            # Step 2: Refine syntax with Gemini
            st.info("Step 2/4: Refining query syntax...")
            refined_cypher = refine_cypher_syntax(draft_cypher, gemini_client, gemini_model)
            
            # Step 3: Entity linking
            st.info("Step 3/4: Linking entities...")
            final_cypher, replacements = replace_entities_in_query(
                refined_cypher, e5_model, entity_index, sim_threshold
            )
            
            # Step 4: Execute query
            st.info("Step 4/4: Executing query and generating answer...")
            results, error = execute_cypher_query(driver, final_cypher)
            
            # Generate final answer
            final_answer = generate_final_answer(
                question, final_cypher, results, error, gemini_client, gemini_model
            )
        
        # Display results
        st.markdown("---")
        
        # Main answer
        st.subheader("Answer")
        st.markdown(final_answer)
        
        # Debug information in expander
        with st.expander("Debug Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Draft Cypher (Local Model)**")
                st.code(draft_cypher, language="cypher")
                
                st.markdown("**Refined Cypher (After Gemini)**")
                st.code(refined_cypher, language="cypher")
            
            with col2:
                st.markdown("**Entity Replacements**")
                if replacements:
                    for r in replacements:
                        status = "REPLACED" if r['replaced'] else "KEPT"
                        st.markdown(
                            f"- `{r['original']}` -> `{r['matched']}` "
                            f"(score: {r['score']:.3f}, {status})"
                        )
                else:
                    st.markdown("No entity terms found in query.")
                
                st.markdown("**Final Cypher Query**")
                st.code(final_cypher, language="cypher")
            
            st.markdown("**Execution Results**")
            if error:
                st.error(f"Error: {error}")
            elif results:
                st.json(results[:10])  # Show first 10 results
                if len(results) > 10:
                    st.info(f"Showing 10 of {len(results)} total results.")
            else:
                st.info("No results returned.")


if __name__ == "__main__":
    main()
