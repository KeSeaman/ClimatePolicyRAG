# Retrieval Experiments: Closing the Vocabulary Gap

## Round 1: 10 Questions

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

The vocabulary gap is only partly and unreliably closed. Next step: a larger eval set with a
dev/test split, and weighting the original query above the expansions (round 2 below).

---

# Round 2: 40 Questions with a Dev/Test Split

**Labels**: 40 questions, 106 relevant documents, labelled from the full text of every
document. The 10 original questions (already inspected) are dev; the 30 new ones were split
10 dev / 20 test by a seeded shuffle before any retrieval ran on them. Two original labels were
revised after the full-text reread (see the note in `eval/gold.json`), which is why the vector
baseline differs from round 1.

**Procedure**: expansions generated once per question (temperature 0) and cached; 85
configurations searched on dev only (candidate pool 5/10/20/40 × RRF constant 10/30/60 ×
original-question weight 1/2/3 for expansion); the best per family fixed; then each scored
once on test.

| Retrieval (settings chosen on dev) | Dev recall / MRR | Test recall / MRR |
|---|---|---|
| vector (baseline) | 0.880 / **0.975** | 0.940 / 0.925 |
| hybrid (pool 10) | 0.866 / 0.967 | 0.940 / **0.963** |
| vector + expansion (pool 40, RRF 10, weight 3) | 0.890 / 0.975 | 0.940 / 0.887 |
| hybrid + expansion (pool 20, RRF 30, weight 2) | **0.905** / 0.925 | **0.963** / 0.925 |

On test, the four configurations disagree on only three questions. Hybrid + expansion's lead
is one question net: it recovers "Which World Bank budget-support loans tie economic reforms to
climate commitments?" (0 → 0.67; the documents say "Development Policy Operation") but drops a
document for the drought-adaptation question. Hybrid's MRR gain on test reverses on dev.

## Findings

-   **Weighting the original question counters the dilution seen in round 1.** Averaged over
    the pool and RRF settings, dev recall for vector + expansion rises from 0.822 (weight 1)
    to 0.850 (weight 2) and 0.866 (weight 3), and for hybrid + expansion from 0.868 to 0.883
    (weight 2; 0.876 at weight 3). Even so, it only brings expansion level with plain vector
    search (0.880).
-   **No configuration is reliably better than vector search here.** Dev and test disagree on
    the ranking, and the differences are one or two documents.
-   **The corpus is the limit, not the method.** With 80 documents, vector search already
    reaches about 0.9 recall@5; the residual misses are umbrella or jargon questions
    ("budget-support loans", "NDC tracking"). Distinguishing retrieval methods needs a corpus
    with many more distractors (hundreds to thousands of documents) and questions written
    without looking at the target documents' wording.

Vector search remains the default. The fusion settings are exposed as
`CLIMATE_RAG_FUSION_POOL`, `CLIMATE_RAG_RRF_K` and `CLIMATE_RAG_ORIGINAL_QUERY_WEIGHT`; if
you enable expansion, use an original-question weight of 2-3.
