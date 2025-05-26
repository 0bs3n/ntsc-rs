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
    fn zero() -> Self;
    fn simd_from_slice(samples: &[Self]) -> Vec<f32> {
        let mut output = Vec::<f32>::new();
        for i in 0..samples.len() * Self::depth() {

        }
        for sample in samples {
            for 
            output.push(sample.to_f32());
        }
        output
    }
    fn convolve(window: &[Self], kernel: &[f32]) -> Self {
        let mut sum = Self::zero();
        for i in 0..window.len() {
            sum = sum.add(window[i].scale_f32(kernel[i]));
        }
        sum 
        // window.iter().zip(kernel)
        // .map(|(&w, &k)| w.scale_f32(k))
        // .reduce(|a, b| a.add(b)).unwrap()
    }

    // #[cfg(feature = "simd")]
    // fn convolve_simd(window: &[Self], kernel: &[f32]) -> Self {
        // use wide::f32x8;
        // let len = window.len();   
        // let mut sum = f32x8::ZERO;

        // let mut i = 0;
        // let chunks = len / 8;

        // for _ in 0..chunks {
            // let window_chunk = f32x8::from(&window[i..i + 4]);
            // let kernel_chunk = f32x8::from(&kernel[i..i + 4]);
            // sum += window_chunk * kernel_chunk; // Element-wise multiply and add
            // i += 8;
        // }

        // let mut total_sum = sum.reduce_add();
        // while i < len {
            // total_sum += window[i].to_f32() * kernel[i];
            // i += 1; 
        // }
        // Self::from_f32(total_sum)
    // }
}
