# Climate Policy RAG System Walkthrough

## 1. Overview
A production-ready Multi-Agent RAG system built with a **polyglot architecture**:
-   **Rust Core (`climate_core`)**: Handles high-performance document parsing and vector search.
-   **Python App (`climate_rag`)**: Orchestrates functional agents using immutable state.

## 2. Architecture

### Rust Core
-   **VectorIndex**: In-memory similarity search using Cosine Similarity.
-   **Document**: Zero-copy data structure exposed to Python via PyO3.
-   **Concurrency**: Uses `rayon` for parallel processing.

### Python Agents
-   **Immutable State**: `AgentState` (Pydantic frozen model) passed through the pipeline.
-   **Pipeline**: Functional composition (`toolz.pipe`) of stateless agent functions:
    1.  `analyze_intent`: Classifies query (Policy vs Scientific).
    2.  `retrieve_documents`: Calls Rust core for context.
    3.  `synthesize_answer`: Generates response (LLM simulation).

## 3. Installation & Build

```bash
# 1. Create Environment
uv venv .venv
source .venv/bin/activate

# 2. Install Dependencies
uv pip install -r pyproject.toml # (implied via maturin)

# 3. Build Rust Core
maturin develop
```

## 4. Usage

Run the end-to-end verification script:

```bash
python run.py
```

### Sample Output
```
Initializing Core...
Ingesting Data...
Running Pipeline...

=== Final Result ===
Query: What are the targets of the Paris Agreement?
Intent: policy
Answer: Based on 3 sources...
Logs:
 - Intent classified as: policy
 - Retrieved 3 docs
 - Synthesis: Generated answer.
```

## 5. Data Ingestion
The system includes modular adapters for real-time data ingestion:
- **ArXiv**: Fetches latest climate policy research papers.
- **World Bank**: Retrieves CO2 emissions data (EN.ATM.CO2E.PC).

Run the ingestion verification:
```bash
python run_ingest.py
```

## 6. Key Files
-   `src/lib.rs`: Rust entrypoint.
-   `climate_rag/ingest/`: Data source adapters.
-   `climate_rag/pipeline.py`: Functional orchestration.
