use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl Document {
    #[new]
    pub fn new(id: String, content: String, metadata: Option<HashMap<String, String>>) -> Self {
        Document {
            id,
            content,
            metadata: metadata.unwrap_or_default(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    #[pyo3(get)]
    pub doc_id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub content_preview: String,
}
