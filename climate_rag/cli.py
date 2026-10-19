"""Command-line entry point: `climate-rag {demo,ingest,ask,eval}` (or `python -m climate_rag`)."""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from climate_rag._core import Document
from climate_rag.config import Settings
from climate_rag.embeddings import make_embedder
from climate_rag.evaluation import (
    GoldQuestion,
    GoldSet,
    render_markdown,
    score_answer,
    summarize,
)
from climate_rag.ingest import DEFAULT_TOPICS, ingest_topics
from climate_rag.knowledge_base import IndexMismatchError, KnowledgeBase, SearchOptions
from climate_rag.llm import LLM, judge_model_name, make_expander, make_judge, make_llm
from climate_rag.models import AgentState
from climate_rag.pipeline import run_rag_pipeline, run_retrieval

logger = logging.getLogger("climate_rag")

SAMPLE_DOCS = [
    Document("sample_paris", "The Paris Agreement sets a goal of holding warming well below "
             "2 degrees Celsius above pre-industrial levels and pursuing efforts to limit it to "
             "1.5 degrees Celsius.", {"source": "UNFCCC", "title": "Paris Agreement, Article 2"}),
    Document("sample_ndc", "Implementation of NDCs requires significant climate finance flows "
             "to developing nations.", {"source": "IPCC", "title": "NDC implementation"}),
    Document("sample_solar", "Solar tracking systems increase efficiency by 25% but require "
             "higher maintenance.", {"source": "TechReport", "title": "Solar trackers"}),
    Document("sample_methane", "Methane has a global warming potential roughly 80 times that of "
             "CO2 over 20 years, so cutting methane slows near-term warming fastest.",
             {"source": "IPCC", "title": "Short-lived climate forcers"}),
]

DEFAULT_INDEX = Path("data/index")
EVAL_INDEX = Path("eval/index")
EVAL_GOLD = Path("eval/gold.json")


def build_kb(settings: Settings, docs: Sequence[Document]) -> KnowledgeBase:
    started = time.perf_counter()
    kb = KnowledgeBase(make_embedder(settings), settings.chunk_chars)
    chunks = kb.add_documents(docs)
    logger.info("Indexed %d chunks from %d documents in %.1fs",
                chunks, len(docs), time.perf_counter() - started)
    return kb


def fetch_corpus(settings: Settings, topics: Sequence[str], limit: int) -> list[Document]:
    logger.info("Fetching %d topics from arXiv and the World Bank...", len(topics))
    docs = asyncio.run(ingest_topics(topics, limit, settings.http_timeout))
    if not docs:
        raise SystemExit("No documents were ingested; check network access (see warnings above).")
    return docs


def load_kb(settings: Settings, path: Path, reembed: bool) -> KnowledgeBase:
    if not (path / "manifest.json").exists():
        raise SystemExit(
            f"No saved index at {path}. Create one with `climate-rag ingest --out {path}`."
        )
    started = time.perf_counter()
    kb = KnowledgeBase.load(path, make_embedder(settings), reembed=reembed)
    logger.info("Loaded %d chunks from %s in %.2fs", len(kb), path, time.perf_counter() - started)
    return kb


def print_result(state: AgentState) -> None:
    print(f"\nQuestion: {state.query}\nIntent:   {state.intent.value}\n")
    print(state.answer, "\n\nSources:")
    for i, c in enumerate(state.contexts, 1):
        print(f"  [{i}] {c.score:.3f}  {c.source:<20} {c.title or c.doc_id}")
        if c.url:
            print(f"       {c.url}")


def expander_for(args: argparse.Namespace, settings: Settings) -> LLM | None:
    return make_expander(settings) if args.expand else None


def search_options(args: argparse.Namespace, settings: Settings) -> SearchOptions:
    return SearchOptions(
        mode=args.retrieval,
        pool=settings.fusion_pool,
        rrf_k=settings.rrf_k,
        original_weight=settings.original_query_weight,
    )


def retrieval_label(args: argparse.Namespace) -> str:
    mode: str = args.retrieval
    return mode + (" + query expansion" if args.expand else "")


def cmd_demo(args: argparse.Namespace, settings: Settings) -> None:
    kb = build_kb(settings, SAMPLE_DOCS)
    print_result(run_rag_pipeline(args.question, kb, make_llm(settings), k=2))


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    topics = args.topic or list(DEFAULT_TOPICS)
    docs = fetch_corpus(settings, topics, args.limit)
    kb = build_kb(settings, docs)
    manifest = kb.save(
        Path(args.out),
        documents=len(docs),
        sources=dict(Counter(d.metadata.get("source", "unknown") for d in docs)),
        topics=topics,
        limit_per_topic=args.limit,
    )
    print(f"Saved {manifest['chunks']} chunks from {len(docs)} documents to {args.out} "
          f"(corpus {manifest['corpus_id']})")


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    if args.index:
        kb = load_kb(settings, Path(args.index), args.reembed)
    else:
        kb = build_kb(settings, fetch_corpus(settings, args.topic or DEFAULT_TOPICS, args.limit))
    state = run_rag_pipeline(args.question, kb, make_llm(settings), args.k,
                             search_options(args, settings), expander_for(args, settings))
    if len(state.search_queries) > 1:
        print("Searched with:", *state.search_queries, sep="\n  ")
    print_result(state)


def cmd_eval(args: argparse.Namespace, settings: Settings) -> None:
    gold = GoldSet.load(Path(args.gold))
    if args.live:
        docs = fetch_corpus(settings, DEFAULT_TOPICS, args.limit)
        kb = build_kb(settings, docs)
        by_source = Counter(d.metadata.get("source", "unknown") for d in docs)
        corpus = (f"live ingest, {len(docs)} documents ("
                  + ", ".join(f"{n} {s}" for s, n in sorted(by_source.items()))
                  + f"), {len(kb)} chunks")
    else:
        kb = load_kb(settings, Path(args.index), args.reembed)
        m = kb.manifest or {}
        sources = ", ".join(f"{n} {s}" for s, n in sorted(m.get("sources", {}).items()))
        corpus = (f"`{args.index}` (corpus `{m.get('corpus_id')}`, built {m.get('created')}): "
                  f"{m.get('documents')} documents ({sources}), {len(kb)} chunks")

    score_retrieval = gold.corpus_id == kb.corpus_id()
    if not score_retrieval:
        logger.warning("Gold labels are for corpus %s, not %s; skipping retrieval metrics.",
                       gold.corpus_id, kb.corpus_id())

    expander = expander_for(args, settings)
    options = search_options(args, settings)
    questions = gold.select(args.split)
    if args.retrieval_only:
        report_retrieval(args, kb, questions, options, expander, score_retrieval)
        return

    llm = make_llm(settings)
    judge = make_judge(settings) if args.judge else None
    judge_name = judge_model_name(settings)

    results = []
    for i, item in enumerate(questions, 1):
        logger.info("Query %d/%d: %s", i, len(questions), item.question)
        started = time.perf_counter()
        state = run_rag_pipeline(item.question, kb, llm, args.k, options, expander)
        elapsed = time.perf_counter() - started
        results.append(score_answer(
            state, elapsed, item.relevant if score_retrieval else (), judge
        ))

    summary = summarize(results)
    judge_line = "off (enable with `--judge`)"
    if args.judge:
        judge_line = f"`{judge_name}`"
        if judge_name == settings.llm_model:
            judge_line += " (same model as the generator, so faithfulness is only indicative)"
    header = [
        f"- **Date**: {datetime.now(UTC):%Y-%m-%d %H:%M UTC}",
        f"- **Embedder**: `{kb.embedder.name}`",
        f"- **LLM**: `{settings.llm_model}` via {settings.llm_base_url}",
        f"- **Retrieval**: {retrieval_label(args)}, k = {args.k}",
        f"- **Judge**: {judge_line}",
        f"- **Corpus**: {corpus}",
        f"- **Questions**: {len(questions)} ({args.split} split)",
        f"- **Retrieval labels**: `{args.gold}`"
        + ("" if score_retrieval else " (not scored: labels are for a different corpus)"),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(results, summary, header))
    output.with_suffix(".json").write_text(json.dumps({
        "summary": summary.model_dump(),
        "settings": {"embedder": kb.embedder.name, "llm": settings.llm_model,
                     "judge": judge_name if args.judge else None, "k": args.k,
                     "retrieval": args.retrieval, "query_expansion": args.expand},
        "corpus": kb.manifest,
        "results": [r.model_dump() for r in results],
    }, indent=2) + "\n")
    print(f"Wrote {output} and {output.with_suffix('.json')}")
    print(summary.model_dump_json(indent=2))


def report_retrieval(
    args: argparse.Namespace,
    kb: KnowledgeBase,
    questions: Sequence[GoldQuestion],
    options: SearchOptions,
    expander: LLM | None,
    labelled: bool,
) -> None:
    """Score retrieval alone (no answers generated) and print a per-question table."""
    if not labelled:
        raise SystemExit("--retrieval-only needs relevance labels for this corpus.")
    results = []
    for item in questions:
        state = run_retrieval(item.question, kb, args.k, options, expander)
        results.append(score_answer(state, 0.0, item.relevant))
    print(f"\nRetrieval: {retrieval_label(args)}, k = {args.k}, "
          f"{len(questions)} questions ({args.split})\n")
    print(f"{'recall':>6} {'RR':>5}  question")
    for r in results:
        assert r.retrieval is not None
        print(f"{r.retrieval.recall:6.2f} {r.retrieval.reciprocal_rank:5.2f}  {r.question}")
    summary = summarize(results)
    print(f"{summary.mean_recall:6.3f} {summary.mrr:5.3f}  mean recall / MRR")


def add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", choices=["vector", "hybrid"], default=None,
                        help="vector, or hybrid (vector + BM25 keywords); default from settings")
    parser.add_argument("--expand", action=argparse.BooleanOptionalAction, default=None,
                        help="also search with LLM-written rephrasings of the question")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="climate-rag", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="answer a question over a few built-in sample documents")
    demo.add_argument("question", nargs="?", default="What are the targets of the Paris Agreement?")
    demo.set_defaults(func=cmd_demo)

    ingest = sub.add_parser("ingest", help="fetch live arXiv + World Bank data and save an index")
    ingest.add_argument("--out", default=str(DEFAULT_INDEX), help=f"default: {DEFAULT_INDEX}")
    ingest.add_argument("--topic", action="append", help="topic to ingest (repeatable)")
    ingest.add_argument("--limit", type=int, default=5, help="documents per source per topic")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="answer a question from a saved index or live data")
    ask.add_argument("question")
    ask.add_argument("--index", help="saved index to query (otherwise ingests live data)")
    ask.add_argument("--topic", action="append", help="live topic to ingest (repeatable)")
    ask.add_argument("--limit", type=int, default=5, help="live documents per source per topic")
    ask.add_argument("-k", type=int, default=None, help="contexts to retrieve")
    ask.add_argument("--reembed", action="store_true",
                     help="re-embed a saved index built with a different embedding model")
    add_retrieval_options(ask)
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="answer and score the gold questions; write a report")
    ev.add_argument("--index", default=str(EVAL_INDEX), help=f"default: {EVAL_INDEX} (frozen)")
    ev.add_argument("--live", action="store_true", help="ingest live data instead of --index")
    ev.add_argument("--gold", default=str(EVAL_GOLD), help="questions + relevance labels")
    ev.add_argument("--judge", action="store_true", help="also score faithfulness with an LLM")
    ev.add_argument("--retrieval-only", action="store_true",
                    help="score retrieval against the labels without generating answers")
    ev.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="which questions to run: dev (for tuning), test, or all")
    ev.add_argument("--output", default="notes/Test_Queries.md")
    ev.add_argument("--limit", type=int, default=5, help="live documents per source per topic")
    ev.add_argument("-k", type=int, default=None, help="contexts to retrieve")
    ev.add_argument("--reembed", action="store_true",
                    help="re-embed a saved index built with a different embedding model")
    add_retrieval_options(ev)
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # httpx logs every request at INFO; keep that for --verbose only.
    logging.getLogger("httpx").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    settings = Settings()
    if getattr(args, "k", None) is None:
        args.k = settings.top_k
    if getattr(args, "retrieval", None) is None:
        args.retrieval = settings.retrieval_mode
    if getattr(args, "expand", None) is None:
        args.expand = settings.query_expansion

    from openai import APIConnectionError, NotFoundError

    try:
        args.func(args, settings)
    except IndexMismatchError as e:
        sys.exit(str(e))
    except APIConnectionError:
        sys.exit(f"Could not reach the LLM at {settings.llm_base_url}. Start Ollama "
                 f"(`ollama serve`) or set CLIMATE_RAG_LLM_BASE_URL.")
    except NotFoundError:
        sys.exit(f"Model {settings.llm_model!r} not found at {settings.llm_base_url}. "
                 f"Run `ollama pull {settings.llm_model}` or set CLIMATE_RAG_LLM_MODEL.")
