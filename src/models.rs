use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass(get_all, set_all, from_py_object, module = "climate_rag._core")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl Document {
    #[new]
    #[pyo3(signature = (id, content, metadata=None))]
    pub fn new(id: String, content: String, metadata: Option<HashMap<String, String>>) -> Self {
        Document {
            id,
            content,
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Document(id={:?}, content_len={})", self.id, self.content.len())
    }
}

#[pyclass(get_all, module = "climate_rag._core")]
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    // Settable so rank fusion in Python can replace it with the fused score.
    #[pyo3(set)]
    pub score: f32,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!("SearchResult(doc_id={:?}, score={:.4})", self.doc_id, self.score)
    }
}
