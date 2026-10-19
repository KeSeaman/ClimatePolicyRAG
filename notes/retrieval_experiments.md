# Retrieval Experiments: Closing the Vocabulary Gap

**Date**: 2026-10-19 · **Corpus**: `eval/index` (`c1c89a67bbb7`, 80 documents, 124 chunks)
· **Labels**: `eval/gold.json` (10 questions, 46 relevant documents) · **k** = 5

## Motivation

With plain vector search, "How are NDCs (Nationally Determined Contributions) being tracked?"
scored recall 0: the relevant documents talk about MRV systems, national inventories and
progress toward pledges, and never say "NDC".

## Methods tried

-   **Hybrid**: BM25 keyword ranking (Rust, `src/keyword.rs`) fused with vector ranking by
    reciprocal rank fusion (RRF, constant 60).
-   **Query expansion**: `gemma3:4b` (temperature 0) writes three alternative phrasings;
    rankings for all four queries are fused with RRF.

Reproduce with `climate-rag eval --retrieval-only --retrieval {vector,hybrid} [--expand]`.

## Results (candidate pool of 20 per ranking, the shipped setting)

| Retrieval | Mean recall@5 | MRR | NDC recall |
|---|---|---|---|
| vector (default) | **0.777** | 0.850 | 0.00 |
| hybrid | 0.723 | **0.900** | 0.00 |
| vector + expansion | 0.680 | 0.850 | **0.25** |
| hybrid + expansion | 0.748 | **0.900** | **0.25** |

Per question: hybrid fixed the methane question (recall 0.67 → 1.00, first relevant document
moved from rank 2 to rank 1) but lost documents on carbon pricing (1.00 → 0.33) and forestry
(1.00 → 0.80). Expansion found MRV and progress-tracking documents for the NDC question, but
with vector search it cut adaptation finance from 0.80 to 0.20.

## Why expansion can hurt: RRF dilution

RRF's standard constant flattens rank differences (rank 1 scores 1/61, rank 10 scores 1/70).
With four queries, a document ranked around 10th in all four lists outscores one ranked 1st
in two lists and absent from the others. For adaptation finance, the expansion "Green Climate
Fund projects" pulled generic World Bank projects that are moderately similar to every
phrasing, and they displaced the specific finance papers.

Varying the per-ranking candidate pool confirms the sensitivity, but shows no reliable winner:

| Retrieval | Pool 5 | Pool 10 | Pool 20 |
|---|---|---|---|
| hybrid | 0.712 | 0.757 | 0.723 |
| vector + expansion | 0.777 | 0.700 | 0.680 |
| hybrid + expansion | 0.793 | 0.743 | 0.748 |

(mean recall@5; NDC recall varied between 0 and 0.25 with no pattern)

## Conclusion

On 10 questions, a difference of 0.05 in mean recall is about two documents, so none of these
configurations is distinguishable from plain vector search, and picking the best cell above
would be fitting the test set. Vector search stays the default; hybrid search and expansion
are opt-in (`--retrieval hybrid`, `--expand`, or the `CLIMATE_RAG_RETRIEVAL_MODE` and
`CLIMATE_RAG_QUERY_EXPANSION` settings). The one consistent signal is that hybrid
configurations put a relevant document first more often (MRR 0.90 vs 0.85), which comes down
to one question.

The vocabulary gap is only partly and unreliably closed. Next steps, in order:

1.  Grow the eval set to roughly 40+ questions with a dev/test split, so fusion settings (pool
    size, RRF constant, extra weight on the original query) can be tuned on dev and confirmed
    on test.
2.  Then try weighting the original query's rankings above the expansions, the most direct
    fix for the dilution above.
3.  Consider expanding with a hypothetical answer passage (HyDE) instead of short queries.
