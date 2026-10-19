use crate::keyword::Bm25;
use crate::models::{Document, SearchResult};
use numpy::ndarray::Dimension;
use numpy::{PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

// Rows below this are scored on one thread; rayon's fork/join overhead dominates otherwise.
const PAR_THRESHOLD: usize = 4096;
const LANES: usize = 8;

const EMBEDDINGS_FILE: &str = "embeddings.f32";
const DOCUMENTS_FILE: &str = "documents.jsonl";

#[pyclass(module = "climate_rag._core")]
pub struct VectorIndex {
    documents: Vec<Document>,
    // Unit-normalised embeddings, row-major (len == documents.len() * dimension),
    // so cosine similarity reduces to a dot product at query time.
    embeddings: Vec<f32>,
    // Document id -> row, so re-adding an id replaces it instead of duplicating it.
    rows: HashMap<String, usize>,
    // BM25 over the same rows; rebuilt from documents on load rather than persisted.
    keywords: Bm25,
    dimension: usize,
}

#[pymethods]
impl VectorIndex {
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("dimension must be > 0"));
        }
        Ok(VectorIndex {
            documents: Vec::new(),
            embeddings: Vec::new(),
            rows: HashMap::new(),
            keywords: Bm25::default(),
            dimension,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Insert a document, replacing any existing document with the same id.
    /// `embedding` may be a list of floats or a 1-D float32 NumPy array.
    pub fn add(&mut self, doc: Document, embedding: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut row = extract_vector(embedding)?;
        self.check_dim(row.len())?;
        normalize(&mut row);
        self.upsert(doc, &row);
        Ok(())
    }

    /// Insert many documents in one call. `embeddings` may be a list of lists or a 2-D
    /// float32 NumPy array, which is read in place without creating Python floats.
    /// All embeddings are validated before anything is inserted.
    pub fn add_batch(&mut self, docs: Vec<Document>, embeddings: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut flat = extract_rows(embeddings, docs.len(), self.dimension)?;
        self.embeddings.reserve(flat.len());
        for (doc, row) in docs.into_iter().zip(flat.chunks_exact_mut(self.dimension)) {
            normalize(row);
            self.upsert(doc, row);
        }
        Ok(())
    }

    /// `query_embedding` may be a list of floats or a 1-D float32 NumPy array.
    pub fn search(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<Vec<SearchResult>> {
        let mut query = extract_vector(query_embedding)?;
        self.check_dim(query.len())?;
        normalize(&mut query);

        // Scoring is pure Rust, so let other Python threads run meanwhile.
        let top = py.detach(|| top_k(&self.embeddings, &query, self.dimension, k));
        Ok(self.results(top))
    }

    /// BM25 keyword search over document content; scores are BM25, not cosine similarity.
    /// Only documents sharing at least one (stemmed, non-stopword) term are returned.
    pub fn keyword_search(&self, py: Python<'_>, query: &str, k: usize) -> Vec<SearchResult> {
        let top = py.detach(|| self.keywords.top_k(query, k));
        self.results(top)
    }

    /// Document ids in insertion order.
    fn ids(&self) -> Vec<String> {
        self.documents.iter().map(|d| d.id.clone()).collect()
    }

    /// Write the index to `dir` as embeddings.f32 (little-endian, unit-normalised rows)
    /// and documents.jsonl (one document per line, same order). Each file is written to a
    /// temporary name and renamed, so an interrupted save never leaves a half-written file.
    pub fn save(&self, py: Python<'_>, dir: PathBuf) -> PyResult<()> {
        py.detach(|| self.write_files(&dir))
    }

    /// Load an index written by `save`. Inconsistent files raise ValueError.
    #[staticmethod]
    pub fn load(py: Python<'_>, dir: PathBuf, dimension: usize) -> PyResult<Self> {
        py.detach(|| read_files(&dir, dimension))
    }

    fn __len__(&self) -> usize {
        self.documents.len()
    }

    fn __contains__(&self, doc_id: &str) -> bool {
        self.rows.contains_key(doc_id)
    }
}

impl VectorIndex {
    fn results(&self, top: Vec<(usize, f32)>) -> Vec<SearchResult> {
        top.into_iter()
            .map(|(i, score)| {
                let doc = &self.documents[i];
                SearchResult {
                    doc_id: doc.id.clone(),
                    score,
                    content: doc.content.clone(),
                    metadata: doc.metadata.clone(),
                }
            })
            .collect()
    }

    fn check_dim(&self, got: usize) -> PyResult<()> {
        if got != self.dimension {
            return Err(dim_mismatch(self.dimension, got));
        }
        Ok(())
    }

    /// `row` must already be normalised.
    fn upsert(&mut self, doc: Document, row: &[f32]) {
        match self.rows.get(&doc.id) {
            Some(&r) => {
                let d = self.dimension;
                self.embeddings[r * d..(r + 1) * d].copy_from_slice(row);
                self.keywords.set(r, &doc.content);
                self.documents[r] = doc;
            }
            None => {
                let r = self.documents.len();
                self.rows.insert(doc.id.clone(), r);
                self.keywords.set(r, &doc.content);
                self.documents.push(doc);
                self.embeddings.extend_from_slice(row);
            }
        }
    }

    fn write_files(&self, dir: &Path) -> PyResult<()> {
        fs::create_dir_all(dir)?;

        let mut embeddings = Vec::with_capacity(self.embeddings.len() * 4);
        for x in &self.embeddings {
            embeddings.extend_from_slice(&x.to_le_bytes());
        }
        write_atomic(&dir.join(EMBEDDINGS_FILE), &embeddings)?;

        let mut documents = Vec::new();
        for doc in &self.documents {
            serde_json::to_writer(&mut documents, doc)
                .map_err(|e| PyValueError::new_err(format!("cannot serialise {:?}: {e}", doc.id)))?;
            documents.push(b'\n');
        }
        write_atomic(&dir.join(DOCUMENTS_FILE), &documents)?;
        Ok(())
    }
}

fn read_files(dir: &Path, dimension: usize) -> PyResult<VectorIndex> {
    let mut index = VectorIndex::new(dimension)?;

    let reader = BufReader::new(File::open(dir.join(DOCUMENTS_FILE))?);
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: Document = serde_json::from_str(&line).map_err(|e| {
            PyValueError::new_err(format!("{DOCUMENTS_FILE} line {}: {e}", i + 1))
        })?;
        let row = index.documents.len();
        if index.rows.insert(doc.id.clone(), row).is_some() {
            return Err(PyValueError::new_err(format!("duplicate document id {:?}", doc.id)));
        }
        index.keywords.set(row, &doc.content);
        index.documents.push(doc);
    }

    let bytes = fs::read(dir.join(EMBEDDINGS_FILE))?;
    let expected = index.documents.len() * dimension * 4;
    if bytes.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{EMBEDDINGS_FILE} has {} bytes but {} documents x {dimension} dims needs {expected}",
            bytes.len(),
            index.documents.len()
        )));
    }
    index.embeddings = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(index)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)
}

fn dim_mismatch(expected: usize, got: usize) -> PyErr {
    PyValueError::new_err(format!(
        "Embedding dimension mismatch. Expected {expected}, got {got}"
    ))
}

/// One memcpy for C-ordered arrays; anything else is copied in logical (row-major) order.
/// `as_slice` alone is not enough: it also succeeds for Fortran-ordered arrays, whose
/// memory order would silently transpose the rows.
fn read_c_order<D: Dimension>(array: &PyReadonlyArray<'_, f32, D>) -> Vec<f32> {
    match array.as_slice() {
        Ok(s) if array.is_c_contiguous() => s.to_vec(),
        _ => array.as_array().iter().copied().collect(),
    }
}

fn extract_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    match obj.extract::<PyReadonlyArray1<'_, f32>>() {
        Ok(array) => Ok(read_c_order(&array)),
        // Lists, tuples and non-float32 arrays go through the generic sequence path.
        Err(_) => obj.extract(),
    }
}

/// Flatten `n` rows of `dim` floats from a 2-D float32 array or a sequence of sequences.
fn extract_rows(obj: &Bound<'_, PyAny>, n: usize, dim: usize) -> PyResult<Vec<f32>> {
    let count_mismatch =
        |got: usize| PyValueError::new_err(format!("Got {n} documents but {got} embeddings"));

    if let Ok(array) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
        let (rows, cols) = (array.shape()[0], array.shape()[1]);
        if rows != n {
            return Err(count_mismatch(rows));
        }
        if cols != dim {
            return Err(dim_mismatch(dim, cols));
        }
        return Ok(read_c_order(&array));
    }

    let rows: Vec<Vec<f32>> = obj.extract()?;
    if rows.len() != n {
        return Err(count_mismatch(rows.len()));
    }
    let mut flat = Vec::with_capacity(n * dim);
    for row in &rows {
        if row.len() != dim {
            return Err(dim_mismatch(dim, row.len()));
        }
        flat.extend_from_slice(row);
    }
    Ok(flat)
}

fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    // Zero vectors are kept as-is and score 0 against everything.
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Dot product with independent accumulators so LLVM can vectorise it
/// (a plain `.sum()` over f32 is a serial dependency chain).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (a_chunks, a_rest) = a.split_at(a.len() - a.len() % LANES);
    let (b_chunks, b_rest) = b.split_at(a_chunks.len());
    for (ca, cb) in a_chunks.chunks_exact(LANES).zip(b_chunks.chunks_exact(LANES)) {
        for l in 0..LANES {
            acc[l] += ca[l] * cb[l];
        }
    }
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

fn top_k(embeddings: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<(usize, f32)> {
    let n = embeddings.len() / dim;
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let score = |(i, row): (usize, &[f32])| (i, dot(query, row));
    let scores: Vec<(usize, f32)> = if n >= PAR_THRESHOLD {
        embeddings.par_chunks_exact(dim).enumerate().map(score).collect()
    } else {
        embeddings.chunks_exact(dim).enumerate().map(score).collect()
    };
    select_top_k(scores, k)
}

/// The k best (row, score) pairs, best first.
pub(crate) fn select_top_k(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }
    // Descending by score, ties broken by insertion order;
    // total_cmp keeps the order total even with NaNs.
    let desc = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    // O(n) selection of the top k, then sort only those k.
    if k < scores.len() {
        scores.select_nth_unstable_by(k - 1, desc);
        scores.truncate(k);
    }
    scores.sort_unstable_by(desc);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_matches_naive_for_all_tail_lengths() {
        for len in 0..40 {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..len).map(|i| 1.0 - i as f32 * 0.25).collect();
            assert!((dot(&a, &b) - naive_dot(&a, &b)).abs() < 1e-3, "len {len}");
        }
    }

    #[test]
    fn normalize_gives_unit_norm_and_keeps_zero_vectors() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut zero = vec![0.0, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }

    #[test]
    fn top_k_matches_full_sort_on_both_code_paths() {
        let dim = 16;
        for n in [10, PAR_THRESHOLD + 7] {
            // Deterministic pseudo-random rows (LCG) so the test needs no extra crates.
            let mut state = 42u32;
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1 << 24) as f32 - 0.5
            };
            let rows: Vec<f32> = (0..n * dim).map(|_| next()).collect();
            let query: Vec<f32> = (0..dim).map(|_| next()).collect();

            let mut expected: Vec<(usize, f32)> = rows
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, r)| (i, dot(&query, r)))
                .collect();
            expected.sort_by(|a, b| b.1.total_cmp(&a.1));

            let got = top_k(&rows, &query, dim, 5);
            let ids = |v: &[(usize, f32)]| v.iter().map(|x| x.0).collect::<Vec<_>>();
            assert_eq!(ids(&got), ids(&expected[..5]), "n = {n}");
        }
    }

    #[test]
    fn top_k_clamps_k_and_breaks_ties_by_insertion_order() {
        let rows = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        assert_eq!(top_k(&rows, &[1.0, 0.0], 2, 10).len(), 3);
        assert!(top_k(&rows, &[1.0, 0.0], 2, 0).is_empty());
        let ids: Vec<usize> = top_k(&rows, &[1.0, 0.0], 2, 2).iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = std::env::temp_dir().join(format!("climate_core_test_{}", std::process::id()));
        let mut index = VectorIndex::new(2).unwrap();
        for (i, row) in [[3.0f32, 4.0], [0.0, 1.0]].iter().enumerate() {
            let mut row = row.to_vec();
            normalize(&mut row);
            index.upsert(Document::new(format!("d{i}"), "text".into(), None), &row);
        }

        index.write_files(&dir).unwrap();
        let loaded = read_files(&dir, 2).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.documents, index.documents);
        assert_eq!(loaded.embeddings, index.embeddings);
        assert_eq!(loaded.rows, index.rows);
    }
}
