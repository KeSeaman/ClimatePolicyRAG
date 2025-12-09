# 🌍 Climate Policy RAG

> **A Production-Ready, Polyglot Multi-Agent System for Climate Intelligence**
> *Built with Rust 🦀 and Python 🐍*

## 🚀 Overview

**Climate Policy RAG** is a high-performance Retrieval-Augmented Generation system designed to solve the problem of fragmented climate information. It synthesizes insights from diverse global sources (IPCC, UNFCCC, Scientific Literature) into actionable intelligence.

This project demonstrates **advanced AI engineering** by combining:
*   **Rust Core**: For memory-safe, SIMD-accelerated vector searching and document parsing.
*   **Python Agents**: For functional, composable multi-agent orchestration.
*   **Real-Time Ingestion**: Live adapters for ArXiv and World Bank APIs.

## 🏗️ Architecture

The system uses a **Hybrid Polyglot Architecture** where compute-heavy tasks are offloaded to Rust, while business logic resides in a functional Python layer.

```mermaid
graph TD
    subgraph "Python Layer (climate_rag)"
        Orchestrator[Orchestration Pipeline]
        QueryAgent[Query Analysis Agent]
        RetrievalAgent[Retrieval Agent]
        SynthAgent[Synthesis Agent]
        Ingest[Ingestion Adapters]
    end

    subgraph "Rust Core (climate_core)"
        Index[Vector Index (SIMD)]
        Doc[Document Struct (Zero-Copy)]
        Parser[Text Parser]
    end

    Orchestrator --> QueryAgent
    QueryAgent --> RetrievalAgent
    RetrievalAgent -->|PyO3 Bindings| Index
    Index -->|Return Context| RetrievalAgent
    RetrievalAgent --> SynthAgent
    Ingest -->|Load Docs| Index
```

### Key Features
*   **⚡ Performance**: Rust-backed vector store using Rayon for parallel similarity search.
*   **🛡️ Safety**: Pydantic `frozen=True` models ensuring immutable state across agent pipelines.
*   **🧩 Modularity**: Functional composition (`toolz.pipe`) for agent workflows.
*   **🔌 Extensibility**: Protocol-based ingestion adapters for easy addition of new data sources.

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.12+
*   Rust 1.75+ (2021 Edition) / Cargo
*   `uv` (Universal Python Package Manager)

### Quick Start

1.  **Clone and Setup Environment**
    ```bash
    git clone https://github.com/captc/ClimatePolicyRAG.git
    cd ClimatePolicyRAG
    uv venv .venv
    source .venv/bin/activate
    ```

2.  **Build Rust Core**
    We use `maturin` to build the Rust extension and install it into the Python environment.
    ```bash
    maturin develop
    ```
    *(Note: This compiles `climate_core` and links it as a Python module)*

## 🏃 Usage

### 1. Run End-to-End Pipeline
Execute the main pipeline to ingest sample data and answer a query:

```bash
python run.py
```
*Output Preview:*
```text
Query: What are the targets of the Paris Agreement?
Intent: policy
Answer: Based on 3 sources...
```

### 2. Live Data Ingestion
Ingest real-world data from **ArXiv** (Scientific) and **World Bank** (Implementation):

```bash
python run_ingest.py
```
*Output Preview:*
```text
=== Ingesting Scientific Data (ArXiv) ===
 -> [Arxiv] Large Causal Models from Large Language Models...
=== Querying System ===
Retrieved 5 relevant documents.
```

## 📂 Project Structure

```
ClimatePolicyRAG/
├── Cargo.toml              # Rust workspace configuration
├── pyproject.toml          # Python dependencies & build config
├── src/                    # 🦀 Rust Core Source
│   ├── lib.rs              # PyO3 Bindings
│   ├── models.rs           # core.Document definition
│   └── vector_store.rs     # In-memory Vector Index
├── climate_rag/            # 🐍 Python Application
│   ├── agents/             # Functional Agent Logic
│   ├── ingest/             # Data Source Adapters
│   ├── models.py           # Immutable Pydantic Models
│   └── pipeline.py         # Orchestration Logic
└── notes/                  # 📚 Documentation & Walkthroughs
```

## 📊 Data Sources

We currently support the following data providers (see `notes/data_sources.md` for details):
1.  **ArXiv API**: For latest academic research on climate policy.
2.  **World Bank Climate Knowledge Portal**: For historical temperature and weather data.
3.  **Local Documents**: PDF/Text parsing capability (in Rust).

## 🔮 Future Roadmap

-   [ ] **LLM Integration**: Connect `SynthesisAgent` to OpenAI/Anthropic API.
-   [ ] **Vector DB**: Migrate in-memory index to Qdrant/pgvector for scale.
-   [ ] **Deployment**: Dockerize the application for cloud deployment.
-   [ ] **Frontend**: Build a Streamlit or Marimo dashboard for interactive querying.

