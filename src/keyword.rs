//! BM25 keyword scoring, kept row-aligned with the vector index so both can rank the same
//! documents. Exact-term matching complements embeddings on acronyms, names and jargon.

use rayon::prelude::*;
use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;
// Rows below this are scored on one thread; rayon's fork/join overhead dominates otherwise.
const PAR_THRESHOLD: usize = 4096;

const STOPWORDS: &[&str] = &[
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "do",
    "does", "for", "from", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on",
    "or", "than", "that", "the", "their", "these", "this", "those", "to", "was", "we", "were",
    "what", "when", "which", "while", "who", "why", "will", "with",
];

#[derive(Default)]
pub struct Bm25 {
    vocab: HashMap<String, u32>,
    // Number of rows containing each term, indexed by term id.
    df: Vec<u32>,
    // Per row: (term id, term frequency), sorted by term id for binary search.
    rows: Vec<Vec<(u32, u32)>>,
    lens: Vec<u32>,
    total_len: u64,
}

impl Bm25 {
    /// Index `text` as `row`, replacing what was there. Rows must be set in order (append)
    /// or replace an existing row.
    pub fn set(&mut self, row: usize, text: &str) {
        let mut counts: HashMap<u32, u32> = HashMap::new();
        let mut len = 0u32;
        for token in tokenize(text) {
            let next_id = self.vocab.len() as u32;
            let id = *self.vocab.entry(token).or_insert(next_id);
            if id as usize == self.df.len() {
                self.df.push(0);
            }
            *counts.entry(id).or_insert(0) += 1;
            len += 1;
        }
        let mut terms: Vec<(u32, u32)> = counts.into_iter().collect();
        terms.sort_unstable();

        if row < self.rows.len() {
            for &(id, _) in &self.rows[row] {
                self.df[id as usize] -= 1;
            }
            self.total_len -= u64::from(self.lens[row]);
        } else {
            self.rows.push(Vec::new());
            self.lens.push(0);
        }
        for &(id, _) in &terms {
            self.df[id as usize] += 1;
        }
        self.total_len += u64::from(len);
        self.rows[row] = terms;
        self.lens[row] = len;
    }

    /// (row, BM25 score) for rows matching at least one query term, best first.
    pub fn top_k(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let n = self.rows.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let mut terms: Vec<(u32, f32)> = tokenize(query)
            .filter_map(|t| self.vocab.get(&t).copied())
            .map(|id| {
                let df = self.df[id as usize] as f32;
                (id, (1.0 + (n as f32 - df + 0.5) / (df + 0.5)).ln())
            })
            .collect();
        terms.sort_unstable_by_key(|t| t.0);
        terms.dedup_by_key(|t| t.0);
        if terms.is_empty() {
            return Vec::new();
        }

        let avg_len = self.total_len as f32 / n as f32;
        let score = |row: usize| -> f32 {
            let norm = K1 * (1.0 - B + B * self.lens[row] as f32 / avg_len.max(1.0));
            terms
                .iter()
                .filter_map(|&(id, idf)| {
                    let doc = &self.rows[row];
                    let tf = doc.binary_search_by_key(&id, |t| t.0).ok().map(|i| doc[i].1)? as f32;
                    Some(idf * tf * (K1 + 1.0) / (tf + norm))
                })
                .sum()
        };
        let scored = |row: usize| Some((row, score(row))).filter(|s| s.1 > 0.0);
        let scores: Vec<(usize, f32)> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter().filter_map(scored).collect()
        } else {
            (0..n).filter_map(scored).collect()
        };
        crate::vector_store::select_top_k(scores, k)
    }
}

/// Lowercased alphanumeric tokens, stopwords removed, lightly stemmed.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .map(stem)
}

/// Strip common English inflections so "tracked", "tracking" and "tracks" share a term.
/// Deliberately conservative: it only has to be consistent between queries and documents.
fn stem(word: String) -> String {
    if !word.is_ascii() || word.len() <= 3 {
        return word;
    }
    if let Some(s) = word.strip_suffix("ies") {
        return format!("{s}y");
    }
    if word.ends_with("sses") {
        return word[..word.len() - 2].to_string();
    }
    for suffix in ["ing", "ed"] {
        if let Some(s) = word.strip_suffix(suffix) {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
    }
    if word.ends_with('s') && !["ss", "us", "is"].iter().any(|e| word.ends_with(e)) {
        return word[..word.len() - 1].to_string();
    }
    word
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_merges_inflections_conservatively() {
        for (word, expected) in [
            ("tracked", "track"), ("tracking", "track"), ("tracks", "track"),
            ("policies", "policy"), ("ndcs", "ndc"), ("emissions", "emission"),
            ("paris", "paris"), ("process", "process"), ("using", "using"), ("gas", "gas"),
        ] {
            assert_eq!(stem(word.to_string()), expected, "{word}");
        }
    }

    #[test]
    fn tokenize_drops_stopwords_and_punctuation() {
        let tokens: Vec<String> = tokenize("How are NDCs (Nationally Determined) tracked?").collect();
        assert_eq!(tokens, ["ndc", "nationally", "determin", "track"]);
    }

    #[test]
    fn rare_terms_outrank_common_ones_and_replacement_updates_stats() {
        let mut bm = Bm25::default();
        bm.set(0, "carbon tax carbon policy");
        bm.set(1, "carbon capture and storage");
        bm.set(2, "methane leak detection");
        let rows = |bm: &Bm25, q: &str| bm.top_k(q, 3).iter().map(|x| x.0).collect::<Vec<_>>();

        assert_eq!(rows(&bm, "carbon storage"), vec![1, 0]); // "storage" is rarer than "carbon"
        assert!(rows(&bm, "unknown words only").is_empty());

        bm.set(1, "methane flaring");
        assert_eq!(rows(&bm, "storage"), Vec::<usize>::new());
        assert_eq!(rows(&bm, "methane"), vec![1, 2]); // shorter row wins on equal tf
        assert_eq!(bm.df[bm.vocab["carbon"] as usize], 1);
    }
}
