use std::{fmt};
use num::complex::Complex64;
use num::Complex;
use std::f64::consts::PI;

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
}

pub enum SamplingMethod {
    Point,
    Linear,
    Bilinear
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
    
    /*
     * Xk = sum(n = 0, N - 1) xn * e^(-i2pikn)/N
    */
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
    
    pub fn fft(&self) -> Signal<Complex64> {
        let _fft = Self::_fft(&self.to_complex64().data, self.data.len());
        Signal::with_data(self.shape, _fft)
    }
    
    /*
     * Xk = Ek + WkN * Ok
     * Xk+N/2 = Ek - WkN * Ok
     */
    fn _fft(data_in: &[Complex64], N: usize) -> Vec<Complex64> {
        let mut X = Vec::<Complex64>::with_capacity(N);
        // SAFETY: N entries being initialized is guaranteed by the structure of the algo
        unsafe { X.set_len(N);}
        // println!("N: {}", N);


        // Once recursed to N == 1, just return the value of the sample.
        // The sum of 1 sample is just that sample's total
        if N == 1 {
            X[0] = data_in[0];
        } else {
            // separate even and odd indices
            let mut Ex = Vec::<Complex64>::with_capacity(N / 2);
            let mut Ox = Vec::<Complex64>::with_capacity(N / 2);

            for m in 0..N / 2 {
                Ex.push(data_in[2 * m]);
                Ox.push(data_in[2 * m + 1]);
            }

            let E = Self::_fft(&Ex, N / 2);
            let O = Self::_fft(&Ox, N / 2);

            for k in 0..N/2 {
                // WkN = e^(-i*2*pi*k)/N
                let W = Complex::new(0.0, (-1.0 * 2.0 * PI * k as f64) / N as f64).exp();
                X[k] = E[k] + O[k] * W;
                X[k + N / 2] = E[k] - O[k] * W;
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
    
    fn read_f64_from_file(file: &str) -> Vec<f64> {
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
        // let data = read_f64_from_file("/home/sen/Projects/RS170/sine_wave.float64");

        let signal = Signal::with_data(
            SignalShape::OneDimensional(data.len()),
            data
        );

        let dft = signal.dft();
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_dft.complex64").unwrap();
        for val in &dft.data {
            f2.write(&val.re.to_le_bytes()).unwrap();
            f2.write(&val.im.to_le_bytes()).unwrap();
        }

        let mag: Vec<f64> = dft.data.iter().map(|x| (x.re.powi(2) + x.im.powi(2)).sqrt()).collect();
        let mut f1 = File::create("/home/sen/Projects/RS170/sine_wave_dft_mag.float64").unwrap();
        for val in mag {
            f1.write(&val.to_le_bytes()).unwrap();
        }
    }
    
    #[test]
    fn it_transforms_fft() {
        let data = read_complex_from_file("/home/sen/Projects/RS170/sine_wave.complex64");
        let signal = Signal::with_data(
            SignalShape::OneDimensional(data.len()),
            data
        );
        let fft = signal.fft();

        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_fft.complex64").unwrap();
        for val in &fft.data {
            f2.write(&val.re.to_le_bytes()).unwrap();
            f2.write(&val.im.to_le_bytes()).unwrap();
        }

        let mag: Vec<f64> = fft.data.iter().map(|x| (x.re.powi(2) + x.im.powi(2)).sqrt()).collect();
        let mut f1 = File::create("/home/sen/Projects/RS170/sine_wave_fft_mag.float64").unwrap();
        for val in mag {
            f1.write(&val.to_le_bytes()).unwrap();
        }
    }
}