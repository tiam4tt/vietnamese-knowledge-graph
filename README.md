# Vietnamese Pharmaceutical Knowledge Graph (VKGPharma)

<div align="center">

[![GitHub License](https://img.shields.io/github/license/tiam4tt/vietnamese-knowledge-graph?style=for-the-badge&color=%23a6e3a1)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-f9e2af?style=for-the-badge&logo=python&logoColor=%23cdd6f4)](https://www.python.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/tiam4tt/vietnamese-knowledge-graph?style=for-the-badge&color=%23b4befe)


</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/palette/macchiato.png" width="400" />
</p>

A natural language question-answering system for Vietnamese pharmaceutical knowledge, built on Neo4j graph database with fine-tuned BartPho model for NLQ-to-Cypher translation.

## 🎯 Overview

VKGPharma is an end-to-end system that enables natural language querying of pharmaceutical knowledge from the Vietnamese Pharmacopoeia. The system combines:

- **Fine-tuned BartPho** model for Vietnamese NLQ-to-Cypher translation
- **Google Gemini AI** for query refinement and answer generation
- **E5 multilingual embeddings** for entity linking
- **Neo4j graph database** for knowledge storage and retrieval
- **Streamlit** web interface for user interaction

## ✨ Key Features

- **Natural Language Querying**: Ask questions in Vietnamese, get accurate answers
- **Hybrid AI Pipeline**: Combines local fine-tuned model with cloud AI for best results
- **Intelligent Entity Linking**: Automatically matches user terms to database entities
- **Knowledge Graph Schema**: 8 entity types, 8 relationship types covering pharmaceutical domain
- 🔍 **Debug Mode**: View intermediate steps in the query pipeline

## 🏗️ System Architecture

### Query Pipeline

The system processes questions through a 5-step pipeline:

1. **Draft Cypher Generation**: Local fine-tuned BartPho model generates initial Cypher query
2. **Syntax Refinement**: Gemini Flash refines syntax errors for Neo4j compatibility
3. **Entity Linking**: E5 embeddings match entity names to actual database IDs
4. **Query Execution**: Execute refined Cypher query on Neo4j database
5. **Answer Generation**: Gemini converts structured results to natural Vietnamese

### Knowledge Graph Schema

**Node Types (8 categories):**
- `DRUG`: Pharmaceutical products, vaccines, biologics
- `CHEMICAL`: Chemical compounds, active ingredients, reagents
- `DISEASE`: Diseases or clinical symptoms
- `ORGANISM`: Bacteria, viruses, cell lines
- `TEST_METHOD`: Testing and verification techniques
- `STANDARD`: Technical specifications (pH, concentration, efficacy)
- `STORAGE_CONDITION`: Environmental storage requirements
- `PRODUCTION_METHOD`: Manufacturing/formulation technologies

**Relationship Types (8 types):**
- `TREATS`: Drug treats Disease
- `CONTAINS`: Drug contains Chemical
- `TARGETS`: Drug targets Organism
- `HAS_STANDARD`: Entity has Standard
- `TESTED_BY`: Drug tested by Method
- `REQUIRES`: Method requires Chemical
- `PRODUCED_BY`: Drug produced by Method
- `STORED_AT`: Drug stored at Condition

## 🛠️ Technology Stack

- **Language**: Python 3.11.14
- **Database**: Neo4j (graph database)
- **Web Framework**: Streamlit
- **Models**:
  - Fine-tuned BartPho (`vinai/bartpho-syllable`) for NLQ-to-Cypher translation
  - Google Gemini 2.5 Flash for refinement and answer generation
  - E5 Multilingual (`intfloat/multilingual-e5-base`) for entity embeddings
- **ML Libraries**: PyTorch, Transformers, Sentence-Transformers

## 📋 Prerequisites

- Python 3.11.14 or higher
- Docker (for Neo4j database)
- Google Gemini API key
- Minimum 8GB RAM (16GB recommended for optimal performance)
- GPU optional (improves model inference speed)

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/vietnamese-knowledge-graph.git
cd vietnamese-knowledge-graph
```

### 2. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Neo4j Database

Run Neo4j in Docker with ports 7474 (HTTP) and 7687 (Bolt):

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v $PWD/data/neo4j:/data \
  -e NEO4J_AUTH=none \
  neo4j:latest
```

Verify Neo4j is running by accessing http://localhost:7474 in your browser.

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
NEO4J_URI=bolt://localhost:7687
GEMINI_API_KEY=your_gemini_api_key_here
```

To obtain a Gemini API key, visit [Google AI Studio](https://aistudio.google.com/app/apikey).

### 6. Build Knowledge Graph

Build the knowledge graph from extracted data:

```bash
python ultils/build_KG.py
```

This will:
- Load graph documents from `data/raw/graph_documents.json`
- Clear and populate the Neo4j database
- Remove orphan nodes
- Generate a visualization

### 7. Build Entity Index

Generate the entity embedding index for entity linking:

```bash
cd ultils
python build_index.py
```

This creates `data/entity_index.pkl` containing pre-computed embeddings for all entities in the database.

### 8. Model Download (Automatic)

The application automatically handles model loading:

- **Local Model**: If a model exists at `models/bartpho-syllable-NLQ2Cypher/`, it will be used
- **Auto-Download**: If no local model is found, the application automatically downloads the fine-tuned model from HuggingFace: [`tiam4tt/bartpho-NLQ-2-cypher-ViPharma`](https://huggingface.co/tiam4tt/bartpho-NLQ-2-cypher-ViPharma)

**Note**: The first run may take a few minutes to download the model (~500MB). Subsequent runs will use the cached model.

## 🐳 Docker Deployment

### Prerequisites

Before using Docker deployment, ensure you have:

- Docker Engine 20.10+ installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose v2.0+ installed (included with Docker Desktop)
- At least 4GB available disk space
- Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Preparation

1. **Create Environment File**

Put your Gemini API key in `.env.example` and rename it to `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

2. **Create Required Directories**

```bash
mkdir -p data/neo4j data/processed data/raw models
```

3. **Place Required Files**

Ensure these files exist:
- `data/raw/graph_documents.json` (for building the knowledge graph)
- `data/entity_index.pkl` (optional - will be auto-generated if missing)

### Using Docker Compose

Docker Compose automatically manages both Neo4j database and the application.

**Start the Services:**

```bash
docker-compose up -d
```

This command will:
- Pull/build necessary Docker images
- Start Neo4j database on ports 7474 (browser) and 7687 (bolt)
- Start the Streamlit application on port 8501
- Mount data volumes for persistence

**Initialize the Knowledge Graph (First Time Only):**

```bash
# Wait 10-15 seconds for Neo4j to fully start, then build the graph
docker-compose exec app python ultils/build_KG.py
```

**Build Entity Index (First Time Only):**

```bash
docker-compose exec app python ultils/build_index.py
```

**Access the Application:**

- **Web Interface**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474

**View Logs:**

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f neo4j
```

**Stop the Services:**

```bash
# Stop and keep data
docker-compose stop

# Stop and remove containers (data persists in volumes)
docker-compose down

# Stop and remove everything including data
docker-compose down -v
```

### Docker Troubleshooting

**Container Won't Start:**

```bash
# Check container logs
docker logs vkgpharma-app
docker logs neo4j

# Check if ports are already in use
netstat -an | grep 8501
netstat -an | grep 7687
```

**Neo4j Connection Issues:**

```bash
# Verify Neo4j is running
docker ps | grep neo4j

# Test Neo4j connectivity
docker exec vkgpharma-app curl -I http://host.docker.internal:7474

# On Linux, use bridge network
docker network inspect bridge
```

**Permission Denied Errors:**

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER data/ models/

# Or run with appropriate user
docker run --user $(id -u):$(id -g) ...
```

**Out of Memory:**

```bash
# Increase Docker memory limit in Docker Desktop settings
# Or limit application memory usage
docker run -m 4g ...
```

**Rebuild After Code Changes:**

```bash
# With Docker Compose
docker-compose up -d --build

# Manual Docker
docker build --no-cache -t vkgpharma-app .
docker stop vkgpharma-app && docker rm vkgpharma-app
# Then run the container again
```

**Clean Up Docker Resources:**

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune
```

## 🎮 Usage

### Running the Application

**Local Development:**
```bash
streamlit run app.py
```

**Docker:**
```bash
docker-compose up -d
```

The application will open in your browser at http://localhost:8501.

### Example Queries

Try these Vietnamese questions:

```
Dược liệu trị được bệnh nào?
(Which diseases can be treated with medicine?)

Thuốc Paracetamol có chứa hoạt chất gì?
(What active ingredients does Paracetamol contain?)

Điều kiện bảo quản thuốc Insulin như thế nào?
(What are the storage conditions for Insulin?)
```

### Configuration Options

- **Similarity Threshold**: Adjust entity linking sensitivity (default: 0.85)
- **Gemini Model**: Choose between Flash (faster) or Pro (more accurate)
- **Debug Mode**: Expand to view intermediate pipeline steps

## 📁 Project Structure

```
vietnamese-knowledge-graph/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker image configuration
├── docker-compose.yml              # Multi-container Docker setup
├── .dockerignore                   # Docker build exclusions
├── .env                            # Environment variables (API keys)
├── .env-example                    # Environment template
├── LICENSE                         # Apache 2.0 license
├── README.md                       # This file
├── data/
│   ├── neo4j/                     # Neo4j database files (Docker volume)
│   │   ├── databases/             # Graph data storage
│   │   ├── transactions/          # Transaction logs
│   │   └── server_id              # Database instance ID
│   ├── processed/                 # Cleaned and processed datasets
│   │   ├── merged_text_final.txt  # Final merged text
│   │   ├── merged_text_final.md   # Markdown version
│   │   ├── merged_text_final_no_breaks.txt
│   │   ├── entities_relations.json # Extracted entities/relations
│   │   └── ViKG-NLQ-2-Cypher-data_cleaned.csv  # Training data
│   ├── raw/                       # Raw extracted data
│   │   ├── graph_documents.json   # Structured graph data
│   │   ├── normalized_texts_segmented_final.csv
│   │   ├── raw_text_extract.csv   # Initial text extraction
│   │   ├── text_chunks.json       # Chunked text data
│   │   ├── ViKG-NLQ-2-Cypher.csv  # Original training data
│   │   └── Duoc-Dien-Viet-Nam-V-tap-2.pdf  # Source document
│   └── entity_index.pkl           # Entity embedding index (generated)
├── models/
│   ├── bartpho-syllable-NLQ2Cypher/  # Fine-tuned NLQ-to-Cypher model
│   │   ├── model.safetensors      # Model weights
│   │   ├── config.json            # Model configuration
│   │   ├── tokenizer_config.json  # Tokenizer settings
│   │   ├── sentencepiece.bpe.model # Tokenizer model
│   │   └── generation_config.json # Generation parameters
│   ├── test_predictions.csv       # Model evaluation results
│   ├── test_results.json          # Test metrics
│   └── __huggingface_repos__.json # HuggingFace cache info
├── notebooks/                      # Data processing pipeline notebooks
│   ├── extract_text.ipynb         # PDF text extraction
│   ├── nomalization-with-LLM.ipynb # Text normalization
│   ├── merge_chunks.ipynb         # Chunk merging
│   ├── get_entities_relations.ipynb # Entity/relation extraction
│   ├── relation_extraction.ipynb  # Relation processing
│   ├── build_KG.ipynb            # Graph construction
│   ├── nlq2cypher-train.ipynb    # Model training
│   ├── error_analysis.ipynb      # Error analysis
│   ├── query_validate.ipynb      # Query validation
│   ├── A_gen_prompt.md           # Answer generation prompt
│   └── Q_gen_prompt.md           # Question generation prompt
├── utils/
│   ├── build_KG.py               # Knowledge graph builder
│   └── build_index.py            # Entity index builder
└── report/
    └── VKGPharma.pdf             # Project documentation
```

## 🔧 Utilities

### Knowledge Graph Builder

[utils/build_KG.py](utils/build_KG.py) loads graph documents and populates the Neo4j database with entities and relationships.

**Usage:**
```bash
python utils/build_KG.py
```

### Entity Index Builder

[utils/build_index.py](utils/build_index.py) extracts all entity names from Neo4j and generates E5 embeddings for entity linking.

**Usage:**
```bash
python utils/build_index.py
```

**Note**: Run this whenever the Neo4j database is updated to refresh the entity index.

## 📊 Data Pipeline

The project includes Jupyter notebooks documenting the complete data processing workflow:

1. **[extract_text.ipynb](notebooks/extract_text.ipynb)**: Extract text from Vietnamese Pharmacopoeia PDFs
2. **[nomalization-with-LLM.ipynb](notebooks/nomalization-with-LLM.ipynb)**: Normalize OCR text using LLM
3. **[merge_chunks.ipynb](notebooks/merge_chunks.ipynb)**: Merge overlapping text chunks
4. **[relation_extraction.ipynb](notebooks/relation_extraction.ipynb)**: Extract entities and relationships
5. **[build_KG.ipynb](notebooks/build_KG.ipynb)**: Construct Neo4j knowledge graph
6. **[nlq2cypher-train.ipynb](notebooks/nlq2cypher-train.ipynb)**: Train NLQ-to-Cypher model

## 🐛 Troubleshooting

### Neo4j Connection Failed

Ensure Docker container is running:
```bash
docker ps | grep neo4j
```

Restart if needed:
```bash
docker restart neo4j
```

### Gemini API Errors

- Verify API key is correct in `.env` file
- Check API quotas at [Google AI Studio](https://aistudio.google.com/)
- Ensure you have billing enabled for production use

### Model Loading Issues

- Verify model files exist in `models/bartpho-syllable-NLQ2Cypher/`
- For GPU issues, check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- If out of memory, close other applications or use CPU-only mode

### Entity Index Missing

Run the index builder:
```bash
cd ultils
python build_index.py
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Vietnamese Pharmacopoeia (Dược điển Việt Nam)** - Source data
- **VinAI Research** - BartPho Vietnamese language model
- **Google** - Gemini API for query refinement
- **Neo4j** - Graph database platform

---

<div align="center">

*Built with ❤️*

</div>