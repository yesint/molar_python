use std::{ffi::c_void, rc::Rc, slice::from_raw_parts, str::FromStr, sync::Arc};

use molar::core::{providers::RandomPosMutProvider, MeasureMasses, OverlappingMut};
use numpy::{
    nalgebra::Const,
    npyffi::{self, npy_intp},
    Ix1, PyArray, PyArray1, ToNpyDims, ToPyArray, PY_ARRAY_API,
};
use pyo3::{ffi::PyObject, prelude::*, types::PyTuple};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
struct Atom( molar::core::Atom );


#[pyclass]
struct Particle {
    // Internal Atom, not directly exposed
    atom: Py<Atom>,

    // Pos as PyArray
    #[pyo3(get,set)]
    pos: Py<numpy::PyArray1<f32>>,

    // Index
    #[pyo3(get)]
    index: usize,
}

#[pymethods]
impl Particle {
    #[getter]
    fn get_name(&self, py: Python) -> String {
        self.atom.borrow(py).0.name.to_string()
    }

    #[setter]
    fn set_name(&mut self, py: Python, val: &str) {
        self.atom.borrow_mut(py).0.name = val.to_owned();
    }

    //pub resname: AsciiString,
    //pub resid: i32, // Could be negative
    //pub resindex: usize,
    // Atom physical properties from topology
    //pub atomic_number: u8,
    //pub mass: f32,
    //pub charge: f32,
    //pub type_name: AsciiString,
    //pub type_id: u32,
    // Specific PDB fields
    //pub chain: AsciiChar,
    //pub bfactor: f32,
    //pub occupancy: f32,
}

#[pyclass]
struct Topology(triomphe::Arc<molar::core::Topology>);

#[pyclass]
struct TopologyU(Option<triomphe::UniqueArc<molar::core::Topology>>);

#[pyclass]
struct State(triomphe::Arc<molar::core::State>);

#[pyclass]
struct StateU(Option<triomphe::UniqueArc<molar::core::State>>);


#[pyclass(unsendable)]
struct FileHandler(molar::io::FileHandler<'static>);

#[pymethods]
impl FileHandler {
    #[staticmethod]
    fn open(fname: &str) -> PyResult<Self> {
        Ok(FileHandler(molar::io::FileHandler::open(fname)?))
    }

    fn read(&mut self) -> PyResult<(TopologyU, StateU)> {
        let (top, st) = self.0.read()?;
        Ok((TopologyU(Some(top)), StateU(Some(st))))
    }
}

#[pyclass(unsendable)]
struct Source (molar::core::Source<molar::core::OverlappingMut>);

#[pymethods]
impl Source {
    #[new]
    fn new_overlapping_mut(topology: &mut TopologyU, state: &mut StateU) -> PyResult<Self> {
        Ok(Source(
            molar::core::Source::<()>::new_overlapping_mut(topology.0.take().unwrap(), state.0.take().unwrap())?
        ))
    }

    fn select_all(&mut self) -> PyResult<Sel> {
        Ok(Sel(self.0.select_all()?))
    }
}

//====================================
#[pyclass(unsendable)]
struct Sel(molar::core::Sel<OverlappingMut>);

#[pymethods]
impl Sel {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn com(&self, _py: Python) -> PyResult<Py<numpy::PyArray1<f32>>> {
        Ok(copy_pos_to_pyarray(_py, &self.0.center_of_mass()?).into_py(_py))
    }

    fn nth_pos(&self, _py: Python, i: usize) -> Py<numpy::PyArray1<f32>> {
        let pos = unsafe { self.0.nth_pos_unchecked_mut(i) };
        map_pos_to_pyarray(_py, pos).into_py(_py)
    }

    /*
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.0 = 0;
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Py<PyAny>> {
        let py = slf.py();
        let cur = slf.0;
        slf.0 += 1;
        if slf.cur < slf.len() {
            let (i,at,pos) = unsafe{slf.sel.nth_unchecked_mut(cur)};
            //Some((i,map_pos_to_pyarray(py,pos)).into_py(py))
            let p = Particle {
                index: i,
                atom: Py::new(py,Atom(at)).unwrap(),
                pos: map_pos_to_pyarray(py,pos).into_py(py),
            };
            Some(p.into_py(py))
        } else {
            None
        }
    }
    */
}

// Constructs PyArray backed by existing Pos data.
fn map_pos_to_pyarray<'py>(py: Python<'py>, data: &mut molar::core::Pos) -> &'py PyArray1<f32> {
    use numpy::Element;
    let mut dims = numpy::ndarray::Dim(3);

    unsafe {
        let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
            f32::get_dtype(py).into_dtype_ptr(),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            std::ptr::null_mut(),                 // no strides
            data.coords.as_mut_ptr() as *mut c_void, // data
            npyffi::NPY_ARRAY_WRITEABLE | npyffi::NPY_ARRAY_F_CONTIGUOUS, // flag
            std::ptr::null_mut(),                    // obj
        );

        PyArray1::<f32>::from_borrowed_ptr(py, ptr)
    }
}

// Constructs new PyArray that copies Pos
fn copy_pos_to_pyarray<'py>(py: Python<'py>, data: &molar::core::Pos) -> &'py PyArray1<f32> {
    unsafe {
        let array = PyArray1::<f32>::new(py, 3, true);
        std::ptr::copy_nonoverlapping(data.coords.as_ptr(), array.data(), 3);
        array
    }
}

//====================================

/// A Python module implemented in Rust.
#[pymodule]
fn molar_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Particle>()?;
    m.add_class::<Topology>()?;
    m.add_class::<State>()?;
    m.add_class::<FileHandler>()?;
    m.add_class::<Source>()?;
    m.add_class::<Sel>()?;
    Ok(())
}
