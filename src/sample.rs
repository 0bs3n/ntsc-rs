use std::fmt::Debug;

pub fn linear_interpolate<S: Sample>(
    x1: f32, y1: &S, 
    x2: f32, y2: &S, 
    x: f32
) -> S {
    if x1 == x2 && x == x1 { 
        return *y1;
    }
    let dist = (x2 - x1).abs();
    let x1_weight = x2 - x;
    let x2_weight = dist - x1_weight;

    y1.scale_f32(x1_weight).add(y2.scale_f32(x2_weight))
}

pub fn bilinear_interpolate<S: Sample>(
    x1: f32, y1: f32, 
    x2: f32, y2: f32, 
    q11: &S, q21: &S, q12: &S, q22: &S, 
    x: f32, y: f32
) -> S {
    let r1 = linear_interpolate(x1, q11, x2, q21, x);
    let r2 = linear_interpolate(x1, q12, x2, q22, x);
    linear_interpolate(y1, &r1, y2, &r2, y)
}

pub trait Sample: Debug + Copy {
    fn scale_f32(&self, scalar: f32) -> Self;
    fn add(&self, other: Self) -> Self;
    fn sub(&self, other: Self) -> Self;
    fn depth() -> usize;
    fn to_f32(&self) -> f32;
    fn to_complex(&self) -> num::complex::Complex64;
    fn zero() -> Self;
    fn magnitude(&self) -> f32;
}
