use pyo3::prelude::*;

pub mod models;
pub mod vector_store;

use models::Document;
use vector_store::VectorIndex;

#[pymodule]
fn climate_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Document>()?;
    m.add_class::<VectorIndex>()?;
    Ok(())
}
