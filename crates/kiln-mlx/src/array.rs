//! `Array`: a safe, owning wrapper over `mlx_array`.
//!
//! Ownership rules (CLAUDE.md FFI discipline):
//! - every `mlx_array_new*` is matched by exactly one `mlx_array_free`,
//!   enforced by `Drop`;
//! - `Clone` goes through `mlx_array_set`, which creates a *new handle*
//!   aliasing the same underlying (internally refcounted) MLX array — cheap,
//!   and both handles are freed independently;
//! - the debug allocation counter tracks every live handle
//!   ([`crate::debug::live_objects`]).
//!
//! MLX is lazy: nothing computes until [`eval`]/[`async_eval`] (or an
//! `item_*`/`data_*` read, which evaluate the array first). Only read back
//! sampled-token-sized outputs; never `item` mid-graph (CLAUDE.md).
//!
//! `Array` is `!Send`/`!Sync` (raw-pointer field): all MLX values live on the
//! engine thread together with the [`crate::Stream`] that produced them.

use crate::error::{MlxError, check};
use crate::{debug, sys};

/// MLX element dtypes used by Kiln (subset of `mlx_dtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Int32,
    Float16,
    Bfloat16,
    Float32,
}

impl Dtype {
    pub(crate) fn to_sys(self) -> sys::mlx_dtype {
        match self {
            Dtype::Bool => sys::mlx_dtype::MLX_BOOL,
            Dtype::Uint8 => sys::mlx_dtype::MLX_UINT8,
            Dtype::Uint16 => sys::mlx_dtype::MLX_UINT16,
            Dtype::Uint32 => sys::mlx_dtype::MLX_UINT32,
            Dtype::Int32 => sys::mlx_dtype::MLX_INT32,
            Dtype::Float16 => sys::mlx_dtype::MLX_FLOAT16,
            Dtype::Bfloat16 => sys::mlx_dtype::MLX_BFLOAT16,
            Dtype::Float32 => sys::mlx_dtype::MLX_FLOAT32,
        }
    }

    pub(crate) fn from_sys(raw: sys::mlx_dtype) -> Option<Self> {
        Some(match raw {
            sys::mlx_dtype::MLX_BOOL => Dtype::Bool,
            sys::mlx_dtype::MLX_UINT8 => Dtype::Uint8,
            sys::mlx_dtype::MLX_UINT16 => Dtype::Uint16,
            sys::mlx_dtype::MLX_UINT32 => Dtype::Uint32,
            sys::mlx_dtype::MLX_INT32 => Dtype::Int32,
            sys::mlx_dtype::MLX_FLOAT16 => Dtype::Float16,
            sys::mlx_dtype::MLX_BFLOAT16 => Dtype::Bfloat16,
            sys::mlx_dtype::MLX_FLOAT32 => Dtype::Float32,
            _ => return None,
        })
    }

    /// Element size in bytes.
    #[allow(unsafe_code)]
    pub fn size(self) -> usize {
        // SAFETY: pure lookup on a valid enum value.
        unsafe { sys::mlx_dtype_size(self.to_sys()) }
    }
}

/// An owning handle to an MLX N-dimensional array.
#[derive(Debug)]
pub struct Array {
    raw: sys::mlx_array,
}

impl Array {
    /// New empty handle for use as an mlx-c output parameter.
    #[allow(unsafe_code)]
    pub(crate) fn new_handle() -> Self {
        crate::init();
        debug::track_new();
        // SAFETY: constructor with no preconditions; freed in Drop.
        Self {
            raw: unsafe { sys::mlx_array_new() },
        }
    }

    /// Scalar float32 array.
    #[allow(unsafe_code)]
    pub fn from_f32(val: f32) -> Self {
        crate::init();
        debug::track_new();
        // SAFETY: constructor with no preconditions; freed in Drop.
        Self {
            raw: unsafe { sys::mlx_array_new_float(val) },
        }
    }

    /// Scalar int32 array.
    #[allow(unsafe_code)]
    pub fn from_i32(val: i32) -> Self {
        crate::init();
        debug::track_new();
        // SAFETY: constructor with no preconditions; freed in Drop.
        Self {
            raw: unsafe { sys::mlx_array_new_int(val) },
        }
    }

    /// Array copied from a `f32` buffer with the given shape.
    pub fn from_f32_slice(data: &[f32], shape: &[i32]) -> Result<Self, MlxError> {
        Self::from_typed(data.as_ptr().cast(), data.len(), shape, Dtype::Float32)
    }

    /// Array copied from a `u32` buffer with the given shape.
    pub fn from_u32_slice(data: &[u32], shape: &[i32]) -> Result<Self, MlxError> {
        Self::from_typed(data.as_ptr().cast(), data.len(), shape, Dtype::Uint32)
    }

    /// Array copied from an `i32` buffer with the given shape.
    pub fn from_i32_slice(data: &[i32], shape: &[i32]) -> Result<Self, MlxError> {
        Self::from_typed(data.as_ptr().cast(), data.len(), shape, Dtype::Int32)
    }

    /// Array copied from a raw little-endian byte buffer of `dtype` elements
    /// — the safetensors loading path (f16/bf16 have no native Rust type).
    pub fn from_raw_bytes(bytes: &[u8], shape: &[i32], dtype: Dtype) -> Result<Self, MlxError> {
        let elem = dtype.size();
        if elem == 0 || !bytes.len().is_multiple_of(elem) {
            return Err(MlxError {
                message: format!(
                    "byte buffer of {} is not a multiple of {elem}-byte {dtype:?} elements",
                    bytes.len()
                ),
            });
        }
        Self::from_typed(bytes.as_ptr().cast(), bytes.len() / elem, shape, dtype)
    }

    #[allow(unsafe_code)]
    fn from_typed(
        data: *const std::ffi::c_void,
        len: usize,
        shape: &[i32],
        dtype: Dtype,
    ) -> Result<Self, MlxError> {
        crate::init();
        let expected: i64 = shape.iter().map(|&d| i64::from(d.max(0))).product();
        if expected != len as i64 {
            return Err(MlxError {
                message: format!("shape {shape:?} wants {expected} elements, buffer has {len}"),
            });
        }
        debug::track_new();
        // SAFETY: `data` points at `len` valid elements of `dtype` (checked
        // against `shape` above); mlx-c copies the buffer.
        let raw = unsafe {
            sys::mlx_array_new_data(data, shape.as_ptr(), shape.len() as i32, dtype.to_sys())
        };
        if raw.ctx.is_null() {
            debug::track_free();
            return Err(MlxError {
                message: "mlx_array_new_data returned a null array".to_owned(),
            });
        }
        Ok(Self { raw })
    }

    pub fn ndim(&self) -> usize {
        #[allow(unsafe_code)]
        // SAFETY: live handle.
        unsafe {
            sys::mlx_array_ndim(self.raw)
        }
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        #[allow(unsafe_code)]
        // SAFETY: live handle.
        unsafe {
            sys::mlx_array_size(self.raw)
        }
    }

    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        if ndim == 0 {
            return Vec::new();
        }
        #[allow(unsafe_code)]
        // SAFETY: mlx_array_shape returns a pointer to `ndim` ints owned by
        // the array, valid while `self` is alive; copied out immediately.
        unsafe {
            std::slice::from_raw_parts(sys::mlx_array_shape(self.raw), ndim).to_vec()
        }
    }

    /// Size of dimension `dim`.
    pub fn dim(&self, dim: i32) -> i32 {
        #[allow(unsafe_code)]
        // SAFETY: live handle; mlx-c bounds-checks and reports errors for a
        // bad `dim` (returning 0 alongside a recorded error).
        unsafe {
            sys::mlx_array_dim(self.raw, dim)
        }
    }

    pub fn dtype(&self) -> Option<Dtype> {
        #[allow(unsafe_code)]
        // SAFETY: live handle.
        Dtype::from_sys(unsafe { sys::mlx_array_dtype(self.raw) })
    }

    /// Forces evaluation of this array (and its pending graph).
    #[allow(unsafe_code)]
    pub fn eval(&self) -> Result<(), MlxError> {
        // SAFETY: live handle.
        check(unsafe { sys::mlx_array_eval(self.raw) })
    }

    /// Reads a scalar f32 (evaluates first).
    #[allow(unsafe_code)]
    pub fn item_f32(&self) -> Result<f32, MlxError> {
        let mut out = 0.0_f32;
        // SAFETY: live handle; out pointer valid for the call.
        check(unsafe { sys::mlx_array_item_float32(&mut out, self.raw) })?;
        Ok(out)
    }

    /// Reads a scalar u32 (evaluates first) — the sampled-token read.
    #[allow(unsafe_code)]
    pub fn item_u32(&self) -> Result<u32, MlxError> {
        let mut out = 0_u32;
        // SAFETY: live handle; out pointer valid for the call.
        check(unsafe { sys::mlx_array_item_uint32(&mut out, self.raw) })?;
        Ok(out)
    }

    /// Copies the evaluated contents out as `u32`. Only valid on
    /// row-contiguous arrays (all Kiln read-back paths are: freshly
    /// evaluated op outputs).
    #[allow(unsafe_code)]
    pub fn data_u32(&self) -> Result<Vec<u32>, MlxError> {
        self.eval()?;
        // SAFETY: live, evaluated handle; pointer checked for null below and
        // read for exactly `size()` elements.
        let ptr = unsafe { sys::mlx_array_data_uint32(self.raw) };
        if ptr.is_null() {
            return Err(MlxError {
                message: "mlx_array_data_uint32 returned null (wrong dtype?)".to_owned(),
            });
        }
        #[allow(unsafe_code)]
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.size()) }.to_vec())
    }

    /// Copies the evaluated contents out as `f32` (see [`Self::data_u32`]).
    #[allow(unsafe_code)]
    pub fn data_f32(&self) -> Result<Vec<f32>, MlxError> {
        self.eval()?;
        // SAFETY: as in `data_u32`.
        let ptr = unsafe { sys::mlx_array_data_float32(self.raw) };
        if ptr.is_null() {
            return Err(MlxError {
                message: "mlx_array_data_float32 returned null (wrong dtype?)".to_owned(),
            });
        }
        #[allow(unsafe_code)]
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.size()) }.to_vec())
    }

    pub(crate) fn raw(&self) -> sys::mlx_array {
        self.raw
    }

    pub(crate) fn raw_out(&mut self) -> *mut sys::mlx_array {
        &mut self.raw
    }

    /// The null `mlx_array` accepted by mlx-c for optional parameters
    /// documented `/* may be null */`.
    pub(crate) fn null_raw() -> sys::mlx_array {
        sys::mlx_array {
            ctx: std::ptr::null_mut(),
        }
    }
}

impl Clone for Array {
    #[allow(unsafe_code)]
    fn clone(&self) -> Self {
        let mut out = Array::new_handle();
        // SAFETY: both handles are live; `mlx_array_set` re-points `out` at
        // the same refcounted MLX array.
        let status = unsafe { sys::mlx_array_set(out.raw_out(), self.raw) };
        // `mlx_array_set` only fails on allocation failure of the handle
        // itself, which new_handle already produced; treat as unreachable
        // without panicking in release.
        debug_assert_eq!(status, 0, "mlx_array_set failed in Clone");
        out
    }
}

impl Drop for Array {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: exactly one free per handle; not used after drop.
        let _ = unsafe { sys::mlx_array_free(self.raw) };
        debug::track_free();
    }
}

/// Internal RAII wrapper for `mlx_vector_array` (eval/concatenate inputs).
pub(crate) struct VectorArray {
    raw: sys::mlx_vector_array,
}

impl VectorArray {
    #[allow(unsafe_code)]
    pub(crate) fn from_arrays(arrays: &[&Array]) -> Result<Self, MlxError> {
        crate::init();
        debug::track_new();
        // SAFETY: constructor with no preconditions; freed in Drop.
        let raw = unsafe { sys::mlx_vector_array_new() };
        let vec = Self { raw };
        for arr in arrays {
            // SAFETY: vector and array handles are live; append copies the
            // (refcounted) array handle into the vector.
            check(unsafe { sys::mlx_vector_array_append_value(vec.raw, arr.raw()) })?;
        }
        Ok(vec)
    }

    pub(crate) fn raw(&self) -> sys::mlx_vector_array {
        self.raw
    }
}

impl Drop for VectorArray {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: exactly one free per `*_new`.
        let _ = unsafe { sys::mlx_vector_array_free(self.raw) };
        debug::track_free();
    }
}

/// Evaluates the given arrays (and everything they depend on), blocking.
#[allow(unsafe_code)]
pub fn eval(outputs: &[&Array]) -> Result<(), MlxError> {
    let vec = VectorArray::from_arrays(outputs)?;
    // SAFETY: live vector handle.
    check(unsafe { sys::mlx_eval(vec.raw()) })
}

/// Schedules evaluation without blocking — the step-boundary pipelining hook
/// (SPEC §7.1 lazy-eval discipline).
#[allow(unsafe_code)]
pub fn async_eval(outputs: &[&Array]) -> Result<(), MlxError> {
    let vec = VectorArray::from_arrays(outputs)?;
    // SAFETY: live vector handle.
    check(unsafe { sys::mlx_async_eval(vec.raw()) })
}
