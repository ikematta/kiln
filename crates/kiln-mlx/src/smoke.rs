//! Phase 0 smoke path: a minimal safe wrapper proving that mlx-c builds,
//! links, and evaluates a lazy graph. Superseded by the real `Array`/`Stream`
//! wrappers in Phase 3 — do not grow this module.

use crate::sys;

/// Adds two scalars on the MLX default CPU stream and returns the evaluated
/// result.
///
/// Exercises the full round trip required by the Phase 0 acceptance test:
/// array creation → lazy `add` → `eval` → item read, with exactly one
/// `*_free` for every `*_new`.
pub fn add_scalars(a: f32, b: f32) -> Result<f32, String> {
    // SAFETY: every handle created below is freed exactly once before
    // returning; mlx-c reports failures via non-zero return codes, which are
    // checked before the result is read.
    #[allow(unsafe_code)]
    unsafe {
        let stream = sys::mlx_default_cpu_stream_new();
        let lhs = sys::mlx_array_new_float(a);
        let rhs = sys::mlx_array_new_float(b);
        let mut sum = sys::mlx_array_new();

        let mut failure: Option<String> = None;
        if sys::mlx_add(&mut sum, lhs, rhs, stream) != 0 {
            failure = Some("mlx_add returned an error".to_string());
        }

        let mut value = 0.0_f32;
        if failure.is_none() {
            let outputs = sys::mlx_vector_array_new_value(sum);
            if sys::mlx_eval(outputs) != 0 {
                failure = Some("mlx_eval returned an error".to_string());
            }
            let _ = sys::mlx_vector_array_free(outputs);
            if failure.is_none() && sys::mlx_array_item_float32(&mut value, sum) != 0 {
                failure = Some("mlx_array_item_float32 returned an error".to_string());
            }
        }

        let _ = sys::mlx_array_free(lhs);
        let _ = sys::mlx_array_free(rhs);
        let _ = sys::mlx_array_free(sum);
        let _ = sys::mlx_stream_free(stream);

        match failure {
            Some(msg) => Err(msg),
            None => Ok(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::add_scalars;

    #[test]
    fn one_plus_two_is_three() {
        assert_eq!(add_scalars(1.0, 2.0).unwrap(), 3.0);
    }
}
