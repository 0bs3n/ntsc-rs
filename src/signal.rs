use std::fs::File;
use std::io::{Error, Write};
use std::{fmt};
use num::complex::Complex64;
use num::Complex;
use std::f64::consts::PI;
use prime_factorization::Factorization;

use crate::sample::Sample;

impl Sample for num::complex::Complex64 {
    fn add(&self, other: Self) -> Self {
        self + other
    }
    fn depth() -> usize {
        1
    }
    fn scale_f32(&self, scalar: f32) -> Self {
        self * Complex::new(scalar as f64, 0.0)
    }
    fn sub(&self, other: Self) -> Self {
        self - other
    }
    fn mul(&self, other: Self) -> Self {
        self * other
    }
    fn to_complex(&self) -> num::complex::Complex64 {
        *self
    }
    fn to_f32(&self) -> f32 {
        unimplemented!()
    }
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    fn magnitude(&self) -> f32 {
        (self.re.powi(2) + self.im.powi(2)).sqrt() as f32
    }
    fn write_to_file(&self, f: &mut File) -> Result<(), std::io::Error> {
        f.write(&self.re.to_le_bytes())?;
        f.write(&self.im.to_le_bytes())?;
        Ok(())
    }
}

pub enum SamplingMethod {
    Point,
    Linear,
    Bilinear
}
#[derive(Copy, Clone, Debug)]
pub enum FftDirection {
    Forward,
    Inverse
}

impl FftDirection {
    fn invert(self) -> Self {
        match self {
            Self::Forward => Self::Inverse,
            Self::Inverse => Self::Forward
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SignalShape {
    // length
    OneDimensional(usize),
    // width, height
    TwoDimensional(usize, usize),
}

impl SignalShape {
    fn radius(&self) -> usize {
        match self {
            SignalShape::OneDimensional(length) => (length - 1) / 2,
            SignalShape::TwoDimensional(width, _) => (width - 1) / 2
        }
    }
}


pub fn moving_average_kernel_new(shape: SignalShape) -> Kernel {
    match shape { 
        SignalShape::OneDimensional(size) => 
            Signal::with_data(shape, vec![1.0 / size as f32; size]),
        SignalShape::TwoDimensional(width, height) => 
            Signal::with_data(shape, vec![1.0 / (width * height) as f32; width * height])
    }
}

enum Radix {
    Two,
    Three,
    Five,
    None
}

impl Radix {
    fn from_factor(factor: u128) -> Self {
        match factor {
            2 => Self::Two,
            3 => Self::Three,
            5 => Self::Five,
            _ => Self::None
        }
    }
}

pub struct RowsIter<'a, S: Sample> {
    data: &'a [S],
    width: usize,
    current_row: usize,
    total_rows: usize
}

impl<'a, S: Sample> Iterator for RowsIter<'a, S> {
    type Item = &'a[S];
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.total_rows {
            let start = self.current_row * self.width;     
            let end = start + self.width;
            self.current_row += 1;
            Some(&self.data[start..end])
        } else {
            None
        }
    }
}

pub struct RowsIterMut<'a, S: Sample> {
    data: &'a mut [S],
    width: usize,
    current_row: usize,
    total_rows: usize
}

impl<'a, S: Sample> Iterator for RowsIterMut<'a, S> {
    type Item = &'a mut [S];
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.total_rows {
            let start = self.current_row * self.width;     
            self.current_row += 1;
            Some(unsafe { 
                std::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(start), self.width)
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Signal<S: Sample> {
    pub shape: SignalShape,
    pub data: Vec<S>
}

pub type Kernel = Signal<f32>;

impl<'a, S: Sample + fmt::Debug> fmt::Display for Signal<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape {
            SignalShape::OneDimensional(_) => write!(f, "{:02x?}", self.data),
            SignalShape::TwoDimensional(width, height) => {
                for y in 0..height {
                    write!(f, "{:02x?}\n", &self.data.get((y * width)..(y * width + width)))?;
                }
                Ok(())
            }
        }
    }
}

impl<'a, S: Sample> Signal<S> {
    pub fn new(shape: SignalShape) -> Self {
        let data_size = match shape {
            SignalShape::OneDimensional(length) => length,
            SignalShape::TwoDimensional(width, height) => width * height
        };
        Signal {
            shape, 
            data: Vec::<S>::with_capacity(data_size)
        }
    }

    pub fn with_data(shape: SignalShape, data: Vec<S>) -> Self {
        Signal {
            shape, 
            data
        }
    }

    pub fn iter_rows(&'a self) -> RowsIter<'a, S> {
        match self.shape {
            SignalShape::OneDimensional(length) => {
                RowsIter {
                    data: &self.data,
                    width: length,
                    current_row: 0,
                    total_rows: 1
                }
            }
            SignalShape::TwoDimensional(width, height) => {
                RowsIter {
                    data: &self.data,
                    width,
                    current_row: 0,
                    total_rows: height
                }
            }
        }
    }

    pub fn iter_rows_mut(&'a mut self) -> RowsIterMut<'a, S> {
        match self.shape {
            SignalShape::OneDimensional(length) => {
                RowsIterMut {
                    data: self.data.as_mut_slice(),
                    width: length,
                    current_row: 0,
                    total_rows: 1
                }
            }
            SignalShape::TwoDimensional(width, height) => {
                RowsIterMut {
                    data: self.data.as_mut_slice(),
                    width,
                    current_row: 0,
                    total_rows: height
                }
            }
        }
    }
    
    pub fn mul(&self, other: Self) -> Self {
        let mut data = Vec::<S>::new();
        for (s, o) in self.data.iter().zip(other.data.iter()) {
            data.push(s.mul(*o)); 
        }
        Signal::with_data(self.shape, data)
    }
    
    /*
     * Xk = sum(n = 0, N - 1) xn * e^(-i2pikn)/N
    */
    #[allow(non_snake_case)]
    pub fn dft(&self) -> Signal<Complex64> {
        let mut X = Signal::<Complex64>::new(self.shape);
        let N = self.data.len();
        for k in 0..N {
            let mut sum = Complex::new(0.0, 0.0);
            for n in 0..N {
                let cn = Complex::new(
                    0.0, 
                    ((-1.0 /*i*/ * 2.0 * PI) * k as f64 * n as f64) / N as f64
                ).exp();
                let xn = self.data[n];
                sum += cn * xn.to_complex();
            }
            X.data.push(sum);
        }
        X
    }
    
    pub fn to_complex64(&self) -> Signal<Complex64> {
        Signal::with_data(self.shape, self.data.iter().map(|x| x.to_complex()).collect()) 
    }
    
    pub fn dft_mag(&self) -> Vec<f64> {
        let dft = self.dft();
        dft.data[0..dft.data.len()].iter().map(|x| (x.re.powi(2) + x.im.powi(2)).sqrt()).collect()
    }
    
    pub fn fft_mixed_radix(&self, factors: &Vec<u128>) {
        for factor in factors {
            match Radix::from_factor(*factor) {
                Radix::Two   => {

                }
                Radix::Three => {

                }
                Radix::Five  => {

                }
                Radix::None  => {
                    
                }
            }
        }
    }
    
    pub fn dump_to_file(&self, filename: &str) -> Result<(), std::io::Error>{
        let mut f = File::create(filename)?;
        for sample in &self.data {
            sample.write_to_file(&mut f)?;
        } 
        Ok(())
    }

    pub fn fft_dyn_alloc_slow(&self) -> Signal<Complex64> {
        let _fft = Self::_fft_dyn_alloc_slow(&self.to_complex64().data, self.data.len());
        Signal::with_data(self.shape, _fft)
    }
    
    pub fn radix2_fft(&self, direction: FftDirection) -> Signal<Complex64> {
        let mut _fft = Vec::<Complex64>::with_capacity(self.data.len());
        for _ in 0..self.data.len() {
            _fft.push(Complex::new(0.0, 0.0));
        }
        unsafe { _fft.set_len(self.data.len()); }
        Self::_radix2_fft(&self.to_complex64().data, &mut _fft, self.data.len(), 1, direction);
        Signal::with_data(self.shape, _fft)
    }
    
    
    #[allow(non_snake_case)]
    fn _radix2_fft(x: &[Complex64], mut X: &mut [Complex64], N: usize, S: usize, D: FftDirection) {
        if N == 1 {
            X[0] = x[0];
        } else {
            let M = N / 2;
            Self::_radix2_fft(&x, &mut X, M, S * 2, D);
            Self::_radix2_fft(&x[S..], &mut X[M..N], M, S * 2, D);

            let sign = match D {
                FftDirection::Forward => -1.0,
                FftDirection::Inverse => 1.0,
            };
        
            for k in 0..M {
                let W = Complex::new(0.0, (sign * 2.0 * PI * k as f64) / N as f64).exp();                 
                let a = X[k];
                let b = X[k + M];
                let A = a + b * W;
                let B = a - b * W;
                X[k] = A;
                X[k + M] = B;
            }
        }

        // TODO: this is ugly, better way to do this?
        if N == x.len() {
            match D {
                FftDirection::Forward => {}
                FftDirection::Inverse => {
                    for k in 0..N {
                        X[k] = X[k] / N as f64;
                    }
                }
            }
        }
    }

    pub fn bluestein_fft(&self, direction: FftDirection) -> Signal<Complex64> {
        let mut _fft = Vec::<Complex64>::with_capacity(self.data.len());
        for _ in 0..self.data.len() {
            _fft.push(Complex::new(0.0, 0.0));
        }
        unsafe { _fft.set_len(self.data.len()); }
        Self::_bluestein_fft(&self.to_complex64().data, &mut _fft, self.data.len(), direction);
        Signal::with_data(self.shape, _fft)
    }
    
    #[allow(non_snake_case)]
    fn _bluestein_fft(x: &[Complex64], X: &mut [Complex64], N: usize, D: FftDirection) {
        let mut a = Vec::<Complex64>::with_capacity(N);
        let mut c = Vec::<Complex64>::with_capacity(N);

        let sign = match D {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0
        };

        for n in 0..N {
            let chirp = Complex::new(0.0, sign * PI / N as f64 * n.pow(2) as f64).exp();
            a.push(x[n] * chirp);

            let chirp_conjugate = Complex::new(0.0, -sign * (PI / N as f64) * n.pow(2) as f64).exp();
            c.push(chirp_conjugate);
        }

        let M = ((2 * N) - 1).next_power_of_two();
        let cz = Complex::new(0.0, 0.0);

        a.resize(M, cz);
        c.resize(M, cz);

        for n in 1..N {
            c[M - n] = c[n];
        }

        let a = Signal::with_data(SignalShape::OneDimensional(M), a);
        let c = Signal::with_data(SignalShape::OneDimensional(M), c);
        
        let fft_a = a.radix2_fft(D);
        let fft_c = c.radix2_fft(D);
        let result = fft_a.mul(fft_c).radix2_fft(D.invert());
        
        let scaling = match D {
            FftDirection::Forward => 1.0,
            FftDirection::Inverse => M as f64 / N as f64 // TODO: maybe just have scaling be an explicit argument to the fft func?
        };

        for k in 0..N {
            X[k] = result.data[k] * Complex::new(0.0, sign * PI / N as f64 * k.pow(2) as f64).exp() * scaling;
        }

    }
    
    /*
     * Xk = Ek + WkN * Ok
     * Xk+N/2 = Ek - WkN * Ok
     */
    #[allow(non_snake_case)]
    fn _fft_dyn_alloc_slow(data_in: &[Complex64], N: usize) -> Vec<Complex64> {
        let mut X = Vec::<Complex64>::with_capacity(N);
        unsafe { X.set_len(N);}

        if N == 1 {
            X[0] = data_in[0];
        } else {
            let mut Xe = Vec::<Complex64>::with_capacity(N / 2);
            let mut Xo = Vec::<Complex64>::with_capacity(N / 2);

            for m in 0..N / 2 {
                Xe.push(data_in[2 * m]);
                Xo.push(data_in[2 * m + 1]);
            }

            let E = Self::_fft_dyn_alloc_slow(&Xe, N / 2);
            let O = Self::_fft_dyn_alloc_slow(&Xo, N / 2);

            for k in 0..N/2 {
                // WkN = e^(-i*2*pi*k)/N
                let W = Complex::new(0.0, (-1.0 * 2.0 * PI * k as f64) / N as f64).exp();
                let a = E[k];
                let b = O[k];
                let A = a + b * W;
                let B = a - b * W;
                X[k] = A;
                X[k + N / 2] = B;
            }
        }
        return X;
    }

    fn window(&self, center: usize, shape: SignalShape, output: &mut Vec<S>) {
        output.clear();
        match self.shape {
            SignalShape::OneDimensional(signal_length) => {
                let c = center as isize;
                let r = shape.radius() as isize;
                let mut fi: usize;
                for i in c - r ..= c + r {
                    if i < 0 { fi = 0 }
                    else if i as usize >= signal_length { fi = signal_length - 1 }
                    else { fi = i as usize }
                    output.push(self.data[fi]);
                }
            }
            SignalShape::TwoDimensional(width, height) => {
                let c = center as isize;
                let yc = (c as usize / width) as isize;
                let xc = (c as usize % width) as isize;
                let r = shape.radius() as isize;
                let mut fy: usize;
                let mut fx: usize;
                for y in yc - r ..= yc + r {
                    for x in xc - r ..= xc + r {
                        if y < 0                     { fy = 0 }
                        else if y as usize >= height { fy = height - 1 }
                        else                         { fy = y as usize }

                        if x < 0                    { fx = 0 }
                        else if x as usize >= width { fx = width - 1 }
                        else                        { fx = x as usize }

                        let i = (fy * width) + fx;

                        output.push(self.data[i]);
                    }
                }
            }
        }
    }

    pub fn filter_in_place(&mut self, kernel: &Kernel) {
        let mut window = Signal::new(kernel.shape);

        for i in 0..self.data.len() {
            self.window(i, kernel.shape, &mut window.data);
            let sample = window.convolve(kernel);
            self.data[i] = sample;
        }
    }

    pub fn filter(&self, kernel: &Kernel) -> Signal<S> {
        let mut output = Vec::<S>::new();
        let mut window = Signal::new(kernel.shape);
        for i in 0..self.data.len() {
            window.data.clear();
            self.window(i, kernel.shape, &mut window.data);
            let sample = window.convolve(kernel);
            // let sample = S::convolve(&window.data, &kernel.data);
            output.push(sample);
        }
        Signal::with_data(kernel.shape, output)
    }
    
    fn convolve(&self, kernel: &Kernel) -> S {
        // TODO: shape and length validation, should match exactly
        let mut output = S::zero();
        for i in 0..self.data.len() {
            output = output.add(self.data[i].scale_f32(kernel.data[i]));
        }
        output
    }

    fn convolve2(&self, kernel: &Signal<S>) -> Signal<S> {
        let mut output = Signal::new(SignalShape::OneDimensional(self.data.len() + kernel.data.len() - 1));
        let kl = kernel.data.len();
        let sl = self.data.len();
        let ol = self.data.len() + kernel.data.len();
        
        let (a, b) = if sl >= kl {
            (self, kernel)
        } else {
            (kernel, self)
        };
        
        let al = a.data.len();
        let bl = b.data.len();
        
        for k in 0..ol {
            let mut sum = S::zero();
            let r = if k < al { 0..k + 1 } else { k - al + 1..al };
            for i in r {
                sum = sum.add(a.data[i].mul(b.data[bl - 1 - i]));
            }
            output.data.push(sum);
        }
        output
    }

    pub fn _resample(&self, scale: f32, sampling_method: SamplingMethod, output: &mut Signal<S>) {
        output.data.clear();
        match self.shape {
            SignalShape::OneDimensional(length) => {
                // TODO: is this correct? Should we leave it at 0?
                let mut x = if scale > 1f32 { 0.5 } else { 0.0 };
                let step = 1.0 / scale;

                while x < length as f32 {
                    match sampling_method {
                        SamplingMethod::Point => {
                            if x == 0.0 {
                                output.data.push(self.data[x as usize]);
                            } else if x >= length as f32 {
                                output.data.push(self.data[length - 1]) 
                            } else {
                                output.data.push(self.data[x as usize]);
                            }
                        }
                        SamplingMethod::Linear => {
                            let x1 = x as usize;
                            let y;
                            if x >= (self.data.len() - 1) as f32 {
                                // linear extrapolation
                                let x0 = (x - 1.0) as usize;
                                let preceeding_val = linear_interpolate(x0 as f32, &self.data[x0], x1 as f32, &self.data[x1], x);
                                let inc = self.data[x1].sub(preceeding_val);
                                y = self.data[x1].sub(inc);
                            } else {
                                let x2 = (x + 1.0) as usize;
                                y = linear_interpolate(x1 as f32, &self.data[x1], x2 as f32, &self.data[x2], x);
                            }
                            output.data.push(y);
                        }
                        _ => panic!("unsupported sampling method")
                    }
                    x += step;
                }
                output.shape = SignalShape::OneDimensional(output.data.len());
            }
            SignalShape::TwoDimensional(width, height) => {
                let new_width  = (width as f32 * scale) as usize;
                let new_height = (height as f32 * scale) as usize;
                match sampling_method {
                    SamplingMethod::Bilinear => {
                        for dy in 0..new_height {
                            let sy = (dy as f32 / (new_height - 1) as f32 ) * (height - 1) as f32;

                            // TODO: avoid ceil() call here
                            if sy.ceil() >= height as f32 { break }

                            for dx in 0..new_width {
                                let sx = (dx as f32 / (new_width - 1) as f32 ) * (width - 1) as f32;
                                let (x1, x2, y1, y2) = bilinear_interpolate_get_neighbors(sx, sy);

                                let q11 = self.data[x1 + (y1 * width)];
                                let q21 = self.data[x2 + (y1 * width)];
                                let q12 = self.data[x1 + (y2 * width)];
                                let q22 = self.data[x2 + (y2 * width)];
                                let p = bilinear_interpolate(x1 as f32, y1 as f32, x2 as f32, y2 as f32, &q11, &q21, &q12, &q22, sx, sy);

                                output.data.push(p);
                            }
                        }
                    }
                    _ => panic!("unsupported sampling method")
                }
                output.shape = SignalShape::TwoDimensional(new_width, new_height);
            }
        }
    }

    pub fn resample(&self, scale: f32, sampling_method: SamplingMethod) -> Signal<S> {
        match self.shape {
            SignalShape::OneDimensional(length) => {
                // TODO: is this correct? Should we leave it at 0?
                let mut x = if scale > 1f32 { 0.5 } else { 0.0 };
                let mut sampled_data = Vec::<S>::new();
                let step = 1.0 / scale;

                /*
                match sampling_method {
                    SamplingMethod::Point => {
                        while x < length as f32 {
                            if x == 0.0 {
                                sampled_data.push(self.data[x as usize]);
                            } else if x >= length as f32 {
                                sampled_data.push(self.data[length - 1]) 
                            } else {
                                sampled_data.push(self.data[x.floor() as usize]);
                            }
                        }
                    }
                    SamplingMethod::Linear => {
                        let sx;
                        if x < 0.0 {
                            sx = 0.0;
                        } else if x >= length as f32 {
                            sx = (length - 1) as f32;
                        } else {
                            sx = x
                        }
                    }
                    _ => panic!("unsupported sampling method")
                }
                */

                while x < length as f32 {
                    match sampling_method {
                        SamplingMethod::Point => {
                            if x == 0.0 {
                                sampled_data.push(self.data[x as usize]);
                            } else if x >= length as f32 {
                                sampled_data.push(self.data[length - 1]) 
                            } else {
                                sampled_data.push(self.data[x as usize]);
                            }
                        }
                        SamplingMethod::Linear => {
                            let x1 = x as usize;
                            let y;
                            if x >= (self.data.len() - 1) as f32 {
                                // linear extrapolation
                                let x0 = (x - 1.0) as usize;
                                let preceeding_val = linear_interpolate(x0 as f32, &self.data[x0], x1 as f32, &self.data[x1], x);
                                let inc = self.data[x1].sub(preceeding_val);
                                y = self.data[x1].sub(inc);
                            } else {
                                let x2 = (x + 1.0) as usize;
                                y = linear_interpolate(x1 as f32, &self.data[x1], x2 as f32, &self.data[x2], x);
                            }
                            sampled_data.push(y);
                        }
                        _ => panic!("unsupported sampling method")
                    }
                    x += step;
                }
                let new_length = sampled_data.len();
                Signal::with_data(SignalShape::OneDimensional(new_length), sampled_data)
            }
            SignalShape::TwoDimensional(width, height) => {
                let new_width  = (width as f32 * scale) as usize;
                let new_height = (height as f32 * scale) as usize;
                let mut sampled = Vec::<S>::with_capacity(new_width * new_height);
                match sampling_method {
                    SamplingMethod::Bilinear => {
                        for dy in 0..new_height {
                            let sy = (dy as f32 / (new_height - 1) as f32 ) * (height - 1) as f32;

                            // TODO: avoid ceil() call here
                            if sy.ceil() >= height as f32 { break }

                            for dx in 0..new_width {
                                let sx = (dx as f32 / (new_width - 1) as f32 ) * (width - 1) as f32;
                                let (x1, x2, y1, y2) = bilinear_interpolate_get_neighbors(sx, sy);

                                // is this functionality useful enough to break out into a
                                // function? it's not quite a window as currently implemented,
                                // since it has no center. Seems like an even width/length window
                                // should be supported though...
                                // on the other hand, window currently instantiates a new object
                                // and this is less heavy
                                let q11 = self.data[x1 + (y1 * width)];
                                let q21 = self.data[x2 + (y1 * width)];
                                let q12 = self.data[x1 + (y2 * width)];
                                let q22 = self.data[x2 + (y2 * width)];
                                let p = bilinear_interpolate(x1 as f32, y1 as f32, x2 as f32, y2 as f32, &q11, &q21, &q12, &q22, sx, sy);

                                sampled.push(p);
                            }
                        }
                    }
                    _ => panic!("unsupported sampling method")
                }

                Signal::with_data(SignalShape::TwoDimensional(new_width, new_height), sampled)
            }
        }
    }
}

// TODO: move kernels to their own compilation module
// TODO: maybe have some pre-computed
// TODO: add implementations for other kernel types
pub fn moving_average_kernel(radius: usize) -> Vec<f32> {
    vec![1.0 / (radius * 2 + 1) as f32; radius * 2 + 1]
}

pub fn moving_average_kernel_new2(radius: usize) -> Kernel {
    let size = radius * 2 + 1;
    let data = vec![1.0 / (radius * 2 + 1) as f32; radius * 2 + 1];
    Signal::with_data(SignalShape::OneDimensional(size), data)
}

pub fn moving_avg_kernel_unsized(kernel: &mut Kernel) {
    match kernel.shape {
        SignalShape::OneDimensional(size) => {
            kernel.data = vec![1.0 / size as f32; size];
        },
        SignalShape::TwoDimensional(width, height) => {
            kernel.data = vec![1.0 / (width * height) as f32; width * height];
        }
    }
}


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

fn bilinear_interpolate_get_neighbors(x: f32, y: f32) -> (usize, usize, usize, usize) {
    // TODO: can we avoid using ceil() here? It's slower than casting
    (x as usize, x.ceil() as usize, y as usize, y.ceil() as usize)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::{BufReader, Read, Write}};

    use super::*;    
    
    fn _read_f64_from_file(file: &str) -> Vec<f64> {
        let mut input = BufReader::new(
            File::open(file).expect("Failed to open file")
        );
        let mut data = Vec::<f64>::new();
        loop {
            use std::io::ErrorKind;
            let mut buffer = [0u8; std::mem::size_of::<f64>()];
            let res = input.read_exact(&mut buffer);
            match res {
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
                _ => {}
            }
            res.expect("Unexpected error during read");
            let val = f64::from_le_bytes(buffer);
            data.push(val);
        }
        data
    }

    fn read_complex_from_file(file: &str) -> Vec<Complex64> {
        let mut input = BufReader::new(
            File::open(file).expect("Failed to open file")
        );
        let mut data = Vec::<Complex64>::new();
        loop {
            use std::io::ErrorKind;
            let mut buffer = [0u8; std::mem::size_of::<f64>()];
            let res = input.read_exact(&mut buffer);
            match res {
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
                _ => {}
            }
            res.expect("Unexpected error during read");
            let re = f64::from_le_bytes(buffer);
            let res = input.read_exact(&mut buffer);
            match res {
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
                _ => {}
            }
            let im = f64::from_le_bytes(buffer);
            data.push(Complex::new(re, im));
        }
        data
    }
    #[test]
    fn it_transforms_dft() {
        let data = read_complex_from_file("/home/sen/Projects/RS170/sine_wave.complex64");
        let signal = Signal::with_data(
            SignalShape::OneDimensional(data.len()),
            data);

        let dft = signal.dft();
        dft.dump_to_file("/home/sen/Projects/RS170/sine_wave_dft.complex64").unwrap();
    }
    
    #[test]
    fn it_transforms_fft_dyn_alloc_slow() {
        let data = read_complex_from_file("/home/sen/Projects/RS170/sine_wave.complex64");
        let signal = Signal::with_data(
            SignalShape::OneDimensional(data.len()),
            data
        );
        let fft = signal.fft_dyn_alloc_slow();
        fft.dump_to_file("/home/sen/Projects/RS170/sine_wave_fft.complex64").unwrap();
    }

    #[test]
    fn it_transforms_fft_radix2() {
        let data = read_complex_from_file("/home/sen/Projects/RS170/sine_wave.complex64");
        let signal = Signal::with_data(
            SignalShape::OneDimensional(data.len()),
            data);
        
        let fft = signal.radix2_fft(FftDirection::Forward);
        fft.dump_to_file("/home/sen/Projects/RS170/sine_wave_fft.complex64");

        let ifft = fft.radix2_fft(FftDirection::Inverse);
        ifft.dump_to_file("/home/sen/Projects/RS170/sine_wave_recovered.complex64").unwrap();
    }
    
    #[test]
    fn it_transforms_fft_bluestein() {
        let data = read_complex_from_file("/home/sen/Projects/RS170/sine_wave.complex64");
        let signal = Signal::with_data(
            SignalShape::OneDimensional(data.len()),
            data);

        let fft = signal.bluestein_fft(FftDirection::Forward);
        fft.dump_to_file("/home/sen/Projects/RS170/sine_wave_fft.complex64").unwrap();

        let ifft = fft.bluestein_fft(FftDirection::Inverse);
        ifft.dump_to_file("/home/sen/Projects/RS170/sine_wave_recovered.complex64").unwrap();
    }
}