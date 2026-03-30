//! SIMD-accelerated element-wise operations on `Vec<T>`.
//!
//! This module provides [`VecExt`], a crate-local trait that extends `Vec<T>`
//! (for scalar types such as `f32` and `f64`) with element-wise SIMD operations.
//! Architecture-specific implementations live under the `arch` module.
//!
//! ## Why a custom trait?
//!
//! Because both `Vec<T>` and the standard operator traits (`std::ops::Add`, etc.)
//! are defined outside this crate, the Rust orphan rule forbids implementing those
//! traits for `Vec<T>` here. `VecExt` is a crate-local trait that provides the
//! same arithmetic without violating the orphan rule.

/// Element-wise SIMD operations for `Vec<T>`, where `T` is a scalar element type
/// such as `f32` or `f64`.
///
/// Because both `Vec<T>` and the standard operator traits (`std::ops::Add`,
/// etc.) are foreign to this crate, the orphan rule prevents implementing them
/// here. `VecExt<T>` is a crate-local trait that provides identical
/// functionality without violating that rule.
///
/// All Vec×Vec methods panic when the two vectors have different lengths.
///
/// Concrete implementations are provided per architecture and scalar type.
pub trait VecExt<T> {
    /// Element-wise addition of two equal-length vectors.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::ops::vec::VecExt;
    /// let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    /// assert_eq!(a.add(&b), vec![9.0f32; 8]);
    /// ```
    fn add(&self, rhs: &Self) -> Vec<T>;

    /// Element-wise subtraction of two equal-length vectors.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn sub(&self, rhs: &Self) -> Vec<T>;

    /// Element-wise multiplication of two equal-length vectors.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::ops::vec::VecExt;
    /// let a = vec![2.0f32; 8];
    /// let b = vec![3.0f32; 8];
    /// assert_eq!(a.mul(&b), vec![6.0f32; 8]);
    /// ```
    fn mul(&self, rhs: &Self) -> Vec<T>;

    /// Element-wise division of two equal-length vectors.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn div(&self, rhs: &Self) -> Vec<T>;

    /// Element-wise remainder (truncated division) of two equal-length vectors.
    ///
    /// Matches Rust's scalar `%` semantics: the result has the same sign as the
    /// dividend. For example, `-7.0 % 3.0` yields `-1.0`, not `2.0`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn rem(&self, rhs: &Self) -> Vec<T>;

    /// Adds `rhs` to every element (scalar broadcast).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::ops::vec::VecExt;
    /// let a = vec![1.0f32; 8];
    /// assert_eq!(a.add_scalar(2.0), vec![3.0f32; 8]);
    /// ```
    fn add_scalar(&self, rhs: T) -> Vec<T>;

    /// Subtracts `rhs` from every element (scalar broadcast).
    fn sub_scalar(&self, rhs: T) -> Vec<T>;

    /// Multiplies every element by `rhs` (scalar broadcast).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::ops::vec::VecExt;
    /// let a = vec![2.0f32; 11];
    /// assert_eq!(a.mul_scalar(3.0), vec![6.0f32; 11]);
    /// ```
    fn mul_scalar(&self, rhs: T) -> Vec<T>;

    /// Divides every element by `rhs` (scalar broadcast).
    fn div_scalar(&self, rhs: T) -> Vec<T>;

    /// In-place element-wise addition: `self[i] += rhs[i]` for all `i`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn add_assign(&mut self, rhs: &Self);

    /// In-place element-wise subtraction: `self[i] -= rhs[i]` for all `i`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn sub_assign(&mut self, rhs: &Self);

    /// In-place element-wise multiplication: `self[i] *= rhs[i]` for all `i`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn mul_assign(&mut self, rhs: &Self);

    /// In-place element-wise division: `self[i] /= rhs[i]` for all `i`.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != rhs.len()`.
    fn div_assign(&mut self, rhs: &Self);

    /// In-place scalar addition: `self[i] += rhs` for all `i`.
    fn add_scalar_assign(&mut self, rhs: T);

    /// In-place scalar subtraction: `self[i] -= rhs` for all `i`.
    fn sub_scalar_assign(&mut self, rhs: T);

    /// In-place scalar multiplication: `self[i] *= rhs` for all `i`.
    fn mul_scalar_assign(&mut self, rhs: T);

    /// In-place scalar division: `self[i] /= rhs` for all `i`.
    fn div_scalar_assign(&mut self, rhs: T);

    /// Returns the sum of all elements.
    ///
    /// Returns the additive identity (e.g. `0.0`) for an empty vector.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::ops::vec::VecExt;
    /// let a = vec![1.0f32; 8];
    /// assert_eq!(a.sum(), 8.0);
    /// ```
    fn sum(&self) -> T;

    /// Returns the product of all elements.
    ///
    /// Returns the multiplicative identity (e.g. `1.0`) for an empty vector.
    fn product(&self) -> T;

    /// Returns the minimum element.
    ///
    /// Returns positive infinity for an empty vector.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use simdmath::ops::vec::VecExt;
    /// let a = vec![3.0f32, 1.0, 2.0];
    /// assert_eq!(a.min(), 1.0);
    /// ```
    fn min(&self) -> T;

    /// Returns the maximum element.
    ///
    /// Returns negative infinity for an empty vector.
    fn max(&self) -> T;
}
