pub fn ulp_diff_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a.signum() != b.signum() {
        return u32::MAX;
    }
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;
    (a_bits.wrapping_sub(b_bits)).unsigned_abs()
}

pub fn ulp_diff_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    if a.signum() != b.signum() {
        return u64::MAX;
    }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits.wrapping_sub(b_bits)).unsigned_abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_diff_f32_exact() {
        assert_eq!(ulp_diff_f32(1.0, 1.0), 0);
        assert_eq!(ulp_diff_f32(f32::NAN, f32::NAN), 0);
        assert_eq!(ulp_diff_f32(f32::NAN, 1.0), u32::MAX);
        assert_eq!(ulp_diff_f32(1.0, -1.0), u32::MAX);
    }

    #[test]
    fn ulp_diff_f64_exact() {
        assert_eq!(ulp_diff_f64(1.0, 1.0), 0);
        assert_eq!(ulp_diff_f64(f64::NAN, f64::NAN), 0);
        assert_eq!(ulp_diff_f64(f64::NAN, 1.0), u64::MAX);
        assert_eq!(ulp_diff_f64(1.0, -1.0), u64::MAX);
    }
}
