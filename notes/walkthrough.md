# Climate Policy RAG: Walkthrough

How a question travels through the system. Setup and usage live in the README.

## 1. Ingestion (`climate_rag/ingest/`)

`ingest_topics(topics)` opens one `httpx.AsyncClient` and fetches every (source, topic) pair
concurrently:

-   **arXiv** (`arxiv.py`): the topic becomes an AND query over its terms
    (`all:carbon AND all:pricing`), sorted by relevance. Requests are serialised 3 s apart, as
    arXiv's API terms ask. Entries missing an id, title or abstract are skipped.
-   **World Bank** (`worldbank.py`): the Projects API (`search.worldbank.org/api/v3/projects`)
    returns project abstracts; each becomes a document headed with country, approval date,
    status and commitment.

Transient failures (timeouts, 429, 5xx) are retried with exponential backoff. A source that
still fails is logged and skipped. Documents are de-duplicated by id.

## 2. Indexing (`knowledge_base.py`, `embeddings.py`, `src/`)

`KnowledgeBase.add_documents` splits documents longer than `chunk_chars` at sentence
boundaries, embeds all chunks in one batch, and calls `VectorIndex.add_batch` in Rust. The
index stores unit-normalised vectors in one contiguous buffer; re-adding an id replaces it.

## 3. Pipeline (`pipeline.py`, `agents/`)

`run_rag_pipeline` composes three pure functions over a frozen `AgentState`:

1.  `analyze_intent`: whole-word keyword scoring into policy / scientific / implementation /
    general.
2.  `retrieve_documents`: embeds the question and takes the top-k chunks, with full text and
    metadata (source, title, URL). Optionally (`--retrieval hybrid`) a BM25 keyword ranking
    from the Rust index is fused with the vector ranking by reciprocal rank fusion, and
    (`--expand`) the LLM writes rephrasings that are searched and fused too.
3.  `synthesize_answer`: sends numbered sources plus intent-specific guidance to the LLM, which
    must cite `[n]` and admit when the sources are insufficient. With no contexts the LLM is not
    called.

Each step appends a timestamped entry to `state.logs`.

## 4. Persistence

`KnowledgeBase.save` has the Rust index write `embeddings.f32` (unit-normalised vectors,
little-endian) and `documents.jsonl`, each via a temp file and rename, then writes
`manifest.json` last with the embedder name, dimension, chunk size, provenance and a
`corpus_id` (hash of the chunk ids). `KnowledgeBase.load` refuses an index whose embedder
differs from the configured one unless `--reembed` is given.

## 5. Evaluation (`evaluation.py`)

`climate-rag eval` runs the questions in `eval/gold.json` against `eval/index` and scores:

-   **Retrieval**: recall@k and reciprocal rank at document level (chunks map back through
    `parent_id`), only when the labels' `corpus_id` matches the index.
-   **Citations**: every `[n]` must refer to a retrieved source.
-   **Ungrounded numbers**: figures in the answer missing from every source it cites.
-   **Faithfulness** (`--judge`): the share of claims an LLM judge marks as supported.

It writes `notes/Test_Queries.md` for reading and `notes/Test_Queries.json` for comparing runs.
