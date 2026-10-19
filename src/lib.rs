use pyo3::prelude::*;

pub mod keyword;
pub mod models;
pub mod vector_store;

use models::{Document, SearchResult};
use vector_store::VectorIndex;

#[pymodule]
#[pyo3(name = "_core")]
fn climate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Document>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<VectorIndex>()?;
    Ok(())
}
