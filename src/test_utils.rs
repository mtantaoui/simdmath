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
