use std::fmt;
use std::marker::PhantomData;
use crate::sample::Sample;
use float_cmp::approx_eq;

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
            Signal::new(shape, 1, vec![1.0 / size as f32; size]),
        SignalShape::TwoDimensional(width, height) => 
            Signal::new(shape, 1, vec![1.0 / (width * height) as f32; width * height])
    }
}

pub struct RowsIter<'a> {
    data: &'a [f32],
    width: usize,
    current_row: usize,
    total_rows: usize
}

impl<'a> Iterator for RowsIter<'a> {
    type Item = &'a[f32];
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

pub struct RowsIterMut<'a> {
    data: &'a mut [f32],
    width: usize,
    current_row: usize,
    total_rows: usize
}

impl<'a> Iterator for RowsIterMut<'a> {
    type Item = &'a mut [f32];
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
pub struct Signal<C: AsMut<[f32]> + AsRef<[f32]>> {
    pub shape: SignalShape,
    pub sample_depth: usize,
    pub data: C,
    _phantom_t: PhantomData<f32>
}

pub type Kernel = Signal<Vec<f32>>;

impl<'a, C: AsMut<[f32]> + AsRef<[f32]> + fmt::Debug> fmt::Display for Signal<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape {
            SignalShape::OneDimensional(_) => write!(f, "{:02x?}", self.data),
            SignalShape::TwoDimensional(width, height) => {
                for y in 0..height {
                    write!(f, "{:02x?}\n", &self.data.as_ref().get((y * width)..(y * width + width)))?;
                }
                Ok(())
            }
        }
    }
}

impl<'a, DataContainer: AsMut<[f32]> + AsRef<[f32]>> Signal<DataContainer> {
    pub fn new(shape: SignalShape, sample_depth: usize, data_container: DataContainer) -> Self {
        Signal {
            shape, 
            sample_depth,
            data: data_container, 
            _phantom_t: PhantomData
        }
    }

    #[inline(never)]
    pub fn from_samples<S: Sample>(
        shape: SignalShape, 
        samples: &[S]) 
    -> Signal<Vec<f32>> {
        Signal {
            shape,
            sample_depth: S::depth(),
            data: S::flatten(samples),
            _phantom_t: PhantomData
        }
    }

    pub fn iter_rows(&'a self) -> RowsIter<'a> {
        match self.shape {
            SignalShape::OneDimensional(length) => {
                RowsIter {
                    data: &self.data.as_ref(),
                    width: length,
                    current_row: 0,
                    total_rows: 1
                }
            }
            SignalShape::TwoDimensional(width, height) => {
                RowsIter {
                    data: &self.data.as_ref(),
                    width,
                    current_row: 0,
                    total_rows: height
                }
            }
        }
    }

    pub fn iter_rows_mut(&'a mut self) -> RowsIterMut<'a> {
        match self.shape {
            SignalShape::OneDimensional(length) => {
                RowsIterMut {
                    data: self.data.as_mut(),
                    width: length,
                    current_row: 0,
                    total_rows: 1
                }
            }
            SignalShape::TwoDimensional(width, height) => {
                RowsIterMut {
                    data: self.data.as_mut(),
                    width,
                    current_row: 0,
                    total_rows: height
                }
            }
        }
    }

    #[inline(never)]
    fn convolve<C: AsRef<[f32]> + AsMut<[f32]>>(&self, kernel: &Signal<C>) -> f32 {
        // TODO: shape and length validation, should match exactly
        let mut output: f32 = 0.0;
        for i in 0..self.data.as_ref().len() {
            output += self.data.as_ref()[i] * kernel.data.as_ref()[i]; 
        }
        output
    }

    fn convolve_simd<C: AsRef<[f32]> + AsMut<[f32]>>(&self, kernel: &Signal<C>) -> f32 {
        use wide::f32x4;
        let len = self.data.as_ref().len();   
        let mut sum = f32x4::ZERO;

        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let window_chunk = f32x4::from(&self.data.as_ref()[i..i + 4]);
            let kernel_chunk = f32x4::from(&kernel.data.as_ref()[i..i + 4]);
            sum += window_chunk * kernel_chunk; // Element-wise multiply and add
            i += 8;
        }

        let mut total_sum = sum.reduce_add();
        while i < len {
            total_sum += self.data.as_ref()[i] * kernel.data.as_ref()[i];
            i += 1; 
        }
        total_sum     
    }

    pub fn expand_kernel(&self, mut kernel: Kernel) -> Kernel {
        if self.sample_depth == 1 { return kernel }
        for i in (0..kernel.data.len() * self.sample_depth).step_by(self.sample_depth) {
            for _ in 0..self.sample_depth {
                kernel.data.insert(i, kernel.data[i]);
            }
        }
        kernel
    }

    #[inline(never)]
    fn window(&self, center: usize, shape: SignalShape, output: &mut Vec<f32>) {
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
                    output.push(self.data.as_ref()[fi]);
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

                        output.push(self.data.as_ref()[i]);
                    }
                }
            }
        }
    }

    pub fn filter_in_place<C: AsRef<[f32]> + AsMut<[f32]>>(&mut self, kernel: &Kernel) {
        let mut window = Signal::new(
            kernel.shape, 
            self.sample_depth, 
            Vec::<f32>::with_capacity(kernel.data.len()));
        for i in 0..self.data.as_ref().len() {
            self.window(i, kernel.shape, &mut window.data);
            let sample = window.convolve(kernel);
            self.data.as_mut()[i] = sample;
        }
    }

    pub fn filter(&self, kernel: &Kernel) -> Signal<Vec<f32>> {
        let mut window = Signal::new(kernel.shape, self.sample_depth, Vec::<f32>::new());
        let mut output = Vec::<f32>::new();
        for i in 0..self.data.as_ref().len() {
            self.window(i, kernel.shape, &mut window.data);
            let sample = window.convolve(&kernel);
            output.push(sample);
        }
        Signal::new(kernel.shape, self.sample_depth, output)
    }

    pub fn _resample(&self, scale: f32, sampling_method: SamplingMethod) -> Signal<Vec<f32>> {
        let mut sampled_data = Vec::<f32>::new();
        let shape: SignalShape;
        let step = 1.0 / scale;
        match self.shape {
            SignalShape::OneDimensional(length) => {
                sampled_data.reserve(length);
                let sample_length = length / self.sample_depth;
                let mut sx: f32 = 0.0;
                match sampling_method {
                    SamplingMethod::Point => {
                        while !approx_eq!(f32, sx, sample_length as f32, ulps = 2) && 
                        sx < sample_length as f32 {
                            for z in 0..self.sample_depth {
                                let ssx = sx as usize * self.sample_depth;
                                let subsample_src_data = self.data.as_ref()[ssx + z];
                                sampled_data.push(subsample_src_data);
                            }
                            sx += step;
                        }
                    }
                    SamplingMethod::Linear => {
                        while !approx_eq!(f32, sx, sample_length as f32, ulps = 2) && 
                        sx < sample_length as f32 {
                            for z in 0..self.sample_depth {
                                let ssx = sx as usize * self.sample_depth;
                                let y = if sx < (sample_length - 1) as f32 {
                                    let x0 = ssx + z;
                                    let x1 = ssx + self.sample_depth + z;
                                    linear_interpolate(x0 as f32, self.data.as_ref()[x0],
                                                       x1 as f32, self.data.as_ref()[x1],
                                                       (sx * self.sample_depth as f32) + z as f32)
                                } else {
                                    let x0 = (ssx - self.sample_depth) + z;
                                    let x1 = ssx + z;
                                    let preceeding_val = 
                                        linear_interpolate(x0 as f32, self.data.as_ref()[x0], 
                                                           x1 as f32, self.data.as_ref()[x1], 
                                                           sx + z as f32);
                                    let inc = self.data.as_ref()[x1] - preceeding_val;
                                    self.data.as_ref()[x1] + inc
                                };
                                sampled_data.push(y);
                            }
                            sx += step;
                        }
                    }
                    _ => panic!()
                }
                shape = SignalShape::OneDimensional(sampled_data.len())
            }
            SignalShape::TwoDimensional(width, height) => {
                let new_width  = (width as f32 * scale) as usize;
                let new_height = (height as f32 * scale) as usize;
                sampled_data.reserve(new_width * new_height);

                #[inline(always)]
                fn get_padded_src_idx(x: usize, y: usize, width: usize, height: usize, sample_depth: usize) -> usize {
                    match (x < width, y < height) {
                        (true, true)   => x + (y * width),
                        (true, false)  => x + ((height - 1) * width),
                        (false, true)  => (width - sample_depth) + (y * width),
                        (false, false) => (width - sample_depth) + ((height - 1) * width),
                    }
                }
                match sampling_method {
                    SamplingMethod::Bilinear => {
                        let mut sy = 0.0;
                        // let mut dy = 0;
                        while !approx_eq!(f32, sy, height as f32, ulps = 2) &&
                            sy < height as f32 {
                            let y1 = sy as usize;
                            let y2 = sy as usize + 1;
                            let mut sx = 0.0;
                            // let mut dx = 0;
                            println!("len: {}", self.data.as_ref().len());
                            while !approx_eq!(f32, sx, width as f32, ulps = 2) 
                                && sx < width as f32 {
                                let x1 = sx as usize;
                                let x2 = sx as usize + self.sample_depth;
                                let i11 = get_padded_src_idx(x1, y1, width, height, self.sample_depth);
                                let i21 = get_padded_src_idx(x2, y1, width, height, self.sample_depth);
                                let i12 = get_padded_src_idx(x1, y2, width, height, self.sample_depth);
                                let i22 = get_padded_src_idx(x2, y2, width, height, self.sample_depth);
                                println!("sx: {sx}, sy: {sy}, i: {i11}, {i21}, {i12}, {i22}");
                                println!("x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}");
                                println!("width: {width}, height: {height}");

                                for z in 0..self.sample_depth {
                                    let q11 = self.data.as_ref()[i11 + z];
                                    let q21 = self.data.as_ref()[i21 + z];
                                    let q12 = self.data.as_ref()[i12 + z];
                                    let q22 = self.data.as_ref()[i22 + z];

                                    let p = bilinear_interpolate(
                                        x1 as f32, y1 as f32, 
                                        x2 as f32, y2 as f32, 
                                        q11, q21, q12, q22, sx, sy);
                                    // println!("p[{subpixel}] = {p}");
                                    sampled_data.push(p);
                                }

                                sx += step * scale;
                                // dx += self.sample_depth;
                            }
                            sy += step;
                            // dy += 1;
                        }
                        // rather than looping by destination x/y, loop by source,
                        // get the values for surrounding pixels for and combo between
                        // a given source x/y, then loop n times where n is scale, with value
                        // i from 0 to 1 in increments of step, passing that as the sx value
                        // to bilinear_interpolate. This avoids redundant calculations of ixx 
                        // and qxx.
                        

                        /*
                        for sy in 0..height {
                            let y1 = sy;
                            let y2 = sy + 1;
                            for sx in (0..width).step_by(self.sample_depth) {
                                let x1 = sx;
                                let x2 = sx + self.sample_depth;
                                let i11 = get_padded_src_idx(x1, y1, width, height, self.sample_depth);
                                let i21 = get_padded_src_idx(x2, y1, width, height, self.sample_depth);
                                let i12 = get_padded_src_idx(x1, y2, width, height, self.sample_depth);
                                let i22 = get_padded_src_idx(x2, y2, width, height, self.sample_depth);
                                let q11 = self.data.as_ref()[i11];
                                let q21 = self.data.as_ref()[i21];
                                let q12 = self.data.as_ref()[i12];
                                let q22 = self.data.as_ref()[i22];

                                let dx = sx as f32;

                                let p = bilinear_interpolate(
                                    x1 as f32, y1 as f32, 
                                    x2 as f32, y2 as f32, 
                                    q11, q21, q12, q22, dx, dy);
                            }
                        }
                        */
                       
                        /*
                        // println!("new_width: {new_width}, new_height: {new_height}");
                        for dy in 0..new_height {
                            // let sy = (dy as f32 / (new_height) as f32 ) * (height) as f32;
                            let sy = dy as f32 * step;

                            // TODO: avoid ceil() call here
                            // if sy.ceil() >= height as f32 { break }
                            let (y1, y2) = (sy as usize, sy as usize + 1);

                            for dx in (0..new_width).step_by(self.sample_depth) {
                                // println!("dx, dy: ({dx}, {dy})");
                                let sx = (dx as f32 * (step / self.sample_depth as f32)) as f32;
                                let (x1, x2) = (sx as usize * self.sample_depth, (sx as usize + 1) * self.sample_depth);
                                // println!("x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}, width: {width}");
                                let i11 = get_padded_src_idx(x1, y1, width, height, self.sample_depth);
                                let i21 = get_padded_src_idx(x2, y1, width, height, self.sample_depth);
                                let i12 = get_padded_src_idx(x1, y2, width, height, self.sample_depth);
                                let i22 = get_padded_src_idx(x2, y2, width, height, self.sample_depth);
                                // println!("sx: {sx}, sy: {sy}, i: {i11}, {i21}, {i12}, {i22}");
                                for i in 0..self.sample_depth {
                                    let i11 = i11 + i;
                                    let i21 = i21 + i;
                                    let i12 = i12 + i;
                                    let i22 = i22 + i;
                                    let q11 = self.data.as_ref()[i11];
                                    let q21 = self.data.as_ref()[i21];
                                    let q12 = self.data.as_ref()[i12];
                                    let q22 = self.data.as_ref()[i22];
                                    // let subpixel = match i {
                                        // 0 => "R",
                                        // 1 => "G",
                                        // 2 => "B",
                                        // _ => panic!()
                                    // };
                                    // println!("q[{subpixel}]: signal[{i11}]={q11}, signal[{i21}]={q21}, signal[{i12}]={q12}, signal[{i22}]={q22}");
                                    let p = bilinear_interpolate(
                                        x1 as f32, y1 as f32, 
                                        x2 as f32, y2 as f32, 
                                        q11, q21, q12, q22, sx, sy);
                                    // println!("p[{subpixel}] = {p}");
                                    sampled_data.push(p);
                                }
                            }
                        }
                       */ 
                    }
                    _ => panic!()
                }
                shape = SignalShape::TwoDimensional(new_width, new_height);
            }
        }
        Signal::new(shape, self.sample_depth, sampled_data)
    }

    pub fn resample(&self, scale: f32, sampling_method: SamplingMethod) -> Signal<Vec<f32>> {
        match self.shape {
            SignalShape::OneDimensional(length) => {
                // Input sample is of subsamples length n, where n is self.sample_depth * input
                // sample length
                
                let mut sampled_data = Vec::<f32>::new();
                // let step = (1.0 / scale) * self.sample_depth as f32;
                let step = (1.0 / scale);
                // let mut x = if scale > 1f32 { 0.5 } else { 0.0 };
                let mut sx = 0.0;
                match sampling_method {
                    SamplingMethod::Point => {
                        // here we iterate over subsamples. for point sampling,
                        // we need to do the following:
                        // for each SAMPLE (subsample group) corresponding to
                        // source sample x ssx, a corresponding SAMPLE (subsample group)
                        // should be written to destination sample x dsx.
                        // 
                        // The iteration is over source subsample x sx. for a signal of sample
                        // length 5, with sample depth 3, this is a subsample length of 15.
                        // step is defined as 1.0 / scale, and scale is a scale applied to the
                        // SUBSAMPLE length. this is fucked I think
                        while sx + (self.sample_depth as f32) < length as f32 {
                            sampled_data.push(self.data.as_ref()[sx as usize]);
                            /*
                            for i in 0..self.sample_depth {
                                println!("source x: {sx}");
                                println!("flat sample idx: {i}");
                                println!("dest x: {}", sx as usize + i);
                                println!("data at dest x: {}", self.data.as_ref()[sx as usize + i]);
                                sampled_data.push(self.data.as_ref()[sx as usize + i])
                            }
                            */
                            // use crate::sample::Sample;
                            // println!("data as rgb: {:?}", 
                                // image::Rgb::<u8>::from_signal_slice(
                                    // &self.data.as_ref()[sx as usize..sx as usize + 3]
                                // ));
                            sx += step * self.sample_depth as f32;
                        }
                    }
                    SamplingMethod::Linear => {
                        while sx + (self.sample_depth as f32) < length as f32 {
                            for i in 0..self.sample_depth {
                                let y = if sx < (length - 1) as f32 {
                                    let x0 = sx as usize + i;
                                    let x1 = (sx + 1.0) as usize + i;
                                    linear_interpolate(
                                        x0 as f32, 
                                        self.data.as_ref()[x0], 
                                        x1 as f32, 
                                        self.data.as_ref()[x1], 
                                        sx)
                                } else {
                                    let x0 = (sx - 1.0) as usize + i;
                                    let x1 = sx as usize + i;
                                    let preceeding_val = linear_interpolate(
                                        x0 as f32, 
                                        self.data.as_ref()[x0], 
                                        x1 as f32, 
                                        self.data.as_ref()[x1], 
                                        sx);
                                    let inc = self.data.as_ref()[x1] - (preceeding_val);
                                    self.data.as_ref()[x1] - (inc)
                                };
                                sampled_data.push(y);
                            }
                            sx += step;
                        }
                    }
                    _ => panic!("unsupported sampling method")
                }
                Signal::new(SignalShape::OneDimensional(sampled_data.len()), self.sample_depth, sampled_data)
            }

            SignalShape::TwoDimensional(width, height) => {
                let new_width  = (width as f32 * scale) as usize;
                let new_height = (height as f32 * scale) as usize;
                let mut sampled = Vec::<f32>::with_capacity(new_width * new_height);
                match sampling_method {
                    SamplingMethod::Bilinear => {
                        for dy in 0..new_height {
                            let sy = (dy as f32 / (new_height - 1) as f32 ) * (height - 1) as f32;

                            // TODO: avoid ceil() call here
                            if sy.ceil() >= height as f32 { break }
                            let (y1, y2) = (sy as usize, sy.ceil() as usize);

                            for dx in (0..new_width * self.sample_depth).step_by(self.sample_depth) {
                                let sx = (dx as f32 * scale) * self.sample_depth as f32;
                                let (x1, x2) = (sx as usize, sx.ceil() as usize);
                                for i in 0..self.sample_depth {
                                    let i11 = match (x1 + i < width, y1 < height) {
                                        (true, true)   => x1 + i + (y1 * width),
                                        (true, false)  => x1 + i + ((height - 1) * width),
                                        (false, true)  => (width - 1) + (y1 * width),
                                        (false, false) => (width - 1) + ((height - 1) * width),
                                    };
                                    let i21 = match (x2 + i < width, y1 < height) {
                                        (true, true)   => x2 + i + (y1 * width),
                                        (true, false)  => x2 + i + ((height - 1) * width),
                                        (false, true)  => (width - 1) + (y1 * width),
                                        (false, false) => (width - 1) + ((height - 1) * width),
                                    };
                                    let i12 = match (x1 + i < width, y2 < height) {
                                        (true, true)   => x1 + i + (y2 * width),
                                        (true, false)  => x1 + i + ((height - 1) * width),
                                        (false, true)  => (width - 1) + (y2 * width),
                                        (false, false) => (width - 1) + ((height - 1) * width),
                                    };
                                    let i22 = match (x2 + i < width, y2 < height) {
                                        (true, true)   => x2 + i + (y2 * width),
                                        (true, false)  => x2 + i + ((height - 1) * width),
                                        (false, true)  => (width - 1) + (y2 * width),
                                        (false, false) => (width - 1) + ((height - 1) * width),
                                    };

                                    let q11 = self.data.as_ref()[i11];
                                    let q21 = self.data.as_ref()[i21];
                                    let q12 = self.data.as_ref()[i12];
                                    let q22 = self.data.as_ref()[i22];
                                    let p = bilinear_interpolate(x1 as f32, y1 as f32, x2 as f32, y2 as f32, q11, q21, q12, q22, sx, sy);

                                    sampled.push(p);
                                }
                            }
                        }
                    }
                    _ => panic!("unsupported sampling method")
                }

                Signal::new(SignalShape::TwoDimensional(new_width, new_height), self.sample_depth, sampled)
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
    Signal::new(SignalShape::OneDimensional(size), 1, data)
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

#[inline(never)]
pub fn linear_interpolate(
    x1: f32, y1: f32, 
    x2: f32, y2: f32, 
    x: f32
) -> f32 {
    if x1 == x2 && x == x1 { 
        return y1;
    }

    // println!("x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}, x: {x}");

    let dist = (x2 - x1).abs();
    // println!("dist: {dist}");
                                
    let x1_weight = (x2 - x) / dist;
    // println!("x1 weight: {x1_weight}");
                            
    let x2_weight = (x1 - x) / dist;
    // println!("x2 weight: {x2_weight}");

    let p = (y1 * x1_weight) + (y2 * x2_weight);
    // println!("p: {p}");

    p
}

pub fn bilinear_interpolate(
    x1: f32, y1: f32, 
    x2: f32, y2: f32, 
    q11: f32, q21: f32, q12: f32, q22: f32, 
    x: f32, y: f32
) -> f32 {
    let r1 = linear_interpolate(x1, q11, x2, q21, x);
    let r2 = linear_interpolate(x1, q12, x2, q22, x);
    linear_interpolate(y1, r1, y2, r2, y)
}

#[inline(never)]
pub fn _linear_interpolate<S: Sample>(
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

pub fn _bilinear_interpolate<S: Sample>(
    x1: f32, y1: f32, 
    x2: f32, y2: f32, 
    q11: &S, q21: &S, q12: &S, q22: &S, 
    x: f32, y: f32
) -> S {
    let r1 = _linear_interpolate(x1, q11, x2, q21, x);
    let r2 = _linear_interpolate(x1, q12, x2, q22, x);
    _linear_interpolate(y1, &r1, y2, &r2, y)
}

fn bilinear_interpolate_get_neighbors(x: f32, y: f32) -> (usize, usize, usize, usize) {
    // TODO: can we avoid using ceil() here? It's slower than casting
    (x as usize, x.ceil() as usize, y as usize, y.ceil() as usize)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Read, Write};
    use super::*;

    fn resample_1d_1m_samples() {
        let mut f = File::open("/home/sen/Projects/RS170/sawtooth.bin").unwrap();
        let mut data = Vec::<u8>::new();
        f.read_to_end(&mut data).unwrap();

        let mut data = data.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let signal = Signal::new(SignalShape::OneDimensional(data.len()), 1, &mut data);
        let resampled = signal.resample(0.1, SamplingMethod::Linear);

        let mut f = File::open("/home/sen/Projects/RS170/sawtooth_downscaled_0.1.bin").unwrap();
        f.write_all(&u8::unflatten(&resampled)).unwrap();
    }
}
