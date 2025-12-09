use crate::models::{Document, SearchResult};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

#[pyclass]
pub struct VectorIndex {
    documents: Arc<RwLock<Vec<Document>>>,
    embeddings: Arc<RwLock<Vec<Vec<f32>>>>,
    dimension: usize,
}

#[pymethods]
impl VectorIndex {
    #[new]
    pub fn new(dimension: usize) -> Self {
        VectorIndex {
            documents: Arc::new(RwLock::new(Vec::new())),
            embeddings: Arc::new(RwLock::new(Vec::new())),
            dimension,
        }
    }

    pub fn add(&mut self, doc: Document, embedding: Vec<f32>) -> PyResult<()> {
        if embedding.len() != self.dimension {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Embedding dimension mismatch. Expected {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        let mut docs = self.documents.write().unwrap();
        let mut embs = self.embeddings.write().unwrap();
        
        docs.push(doc);
        embs.push(embedding);
        
        Ok(())
    }

    pub fn search(&self, query_embedding: Vec<f32>, k: usize) -> PyResult<Vec<SearchResult>> {
        if query_embedding.len() != self.dimension {
             return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
        }

        let docs = self.documents.read().unwrap();
        let embs = self.embeddings.read().unwrap();

        // Calculate cosine similarity
        let mut scores: Vec<(usize, f32)> = embs.par_iter().enumerate()
            .map(|(i, vec)| {
                let score = cosine_similarity(&query_embedding, vec);
                (i, score)
            })
            .collect();

        // Sort descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K
        let results = scores.into_iter()
            .take(k)
            .map(|(i, score)| {
                let doc = &docs[i];
                SearchResult {
                    doc_id: doc.id.clone(),
                    score,
                    content_preview: doc.content.chars().take(100).collect(),
                }
            })
            .collect();

        Ok(results)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
