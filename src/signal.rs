use std::fmt;
use std::marker::PhantomData;
use crate::sample::Sample;

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
    /*
    fn len(&self) -> usize {
        match self {
            SignalShape::OneDimensional(length) => *length,
            SignalShape::TwoDimensional(width, height) => width * height
        }
    }
    */
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
            Signal::new(shape, vec![1.0 / size as f32; size]),
        SignalShape::TwoDimensional(width, height) => 
            Signal::new(shape, vec![1.0 / (width * height) as f32; width * height])
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
pub struct Signal<S: Sample, C: AsMut<[S]> + AsRef<[S]>> {
    pub shape: SignalShape,
    pub data: C,
    _phantom_t: PhantomData<S>
}

pub type Kernel = Signal<f32, Vec<f32>>;

impl<'a, S: Sample, C: AsMut<[S]> + AsRef<[S]> + fmt::Debug> fmt::Display for Signal<S, C> {
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

impl<'a, S: Sample, DataContainer: AsMut<[S]> + AsRef<[S]>> Signal<S, DataContainer> {
    pub fn new(shape: SignalShape, data_container: DataContainer) -> Self {
        Signal {
            shape, 
            data: data_container, 
            _phantom_t: PhantomData
        }
    }

    pub fn iter_rows(&'a self) -> RowsIter<'a, S> {
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

    pub fn iter_rows_mut(&'a mut self) -> RowsIterMut<'a, S> {
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

    fn _window<W>(&self, center: usize, window: &mut Signal<S, W>)
    where
        W: AsRef<[S]> + AsMut<[S]>
    {
        match self.shape {
            SignalShape::OneDimensional(signal_length) => {
                let c = center as isize;
                let r = window.shape.radius() as isize;
                let mut wi = 0;
                let mut fi: usize;
                for i in c - r ..= c + r {
                    if i < 0 { fi = 0 }
                    else if i as usize >= signal_length { fi = signal_length - 1 }
                    else { fi = i as usize }
                    window.data.as_mut()[wi] = self.data.as_ref()[fi];
                    wi += 1;
                }
            }
            SignalShape::TwoDimensional(width, height) => {
                let c = center as isize;
                let yc = (c as usize / width) as isize;
                let xc = (c as usize % width) as isize;
                let r = window.shape.radius() as isize;
                let mut wi = 0;
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

                        window.data.as_mut()[wi] = self.data.as_ref()[i];
                        wi += 1;
                    }
                }
            }
        }
    }

    fn window(&self, center: usize, shape: SignalShape) -> Signal<S, Vec<S>> {
        let mut output = Vec::<S>::new();
       
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
        Signal::new(shape, output)
    }

    pub fn filter_in_place(&mut self, kernel: &Kernel) {
        let mut window = Signal::new(kernel.shape, Vec::<S>::with_capacity(kernel.data.len()));

        // SAFETY: self._window is guaranteed initialize the full vec
        // (in the future anyway, right now this is not safe lol)
        unsafe { window.data.set_len(kernel.data.len()) }

        for i in 0..self.data.as_ref().len() {
            self._window(i, &mut window);
            let sample = S::convolve(&window.data, &kernel.data);
            self.data.as_mut()[i] = sample;
        }
    }

    pub fn filter(&self, kernel: &Kernel) -> Signal<S, Vec<S>> {
        let mut output = Vec::<S>::new();
        for i in 0..self.data.as_ref().len() {
            let sample = S::convolve(
                self.window(i, kernel.shape).data.as_slice(), 
                kernel.data.as_slice());
            output.push(sample);
        }
        Signal::new(kernel.shape, output)
    }

    pub fn _resample(&self, scale: f32, sampling_method: SamplingMethod, output: &mut Signal<S, Vec<S>>) {
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
                                output.data.push(self.data.as_ref()[x as usize]);
                            } else if x >= length as f32 {
                                output.data.push(self.data.as_ref()[length - 1]) 
                            } else {
                                output.data.push(self.data.as_ref()[x as usize]);
                            }
                        }
                        SamplingMethod::Linear => {
                            let x1 = x as usize;
                            let y;
                            if x >= (self.data.as_ref().len() - 1) as f32 {
                                // linear extrapolation
                                let x0 = (x - 1.0) as usize;
                                let preceeding_val = linear_interpolate(x0 as f32, &self.data.as_ref()[x0], x1 as f32, &self.data.as_ref()[x1], x);
                                let inc = self.data.as_ref()[x1].sub(preceeding_val);
                                y = self.data.as_ref()[x1].sub(inc);
                            } else {
                                let x2 = (x + 1.0) as usize;
                                y = linear_interpolate(x1 as f32, &self.data.as_ref()[x1], x2 as f32, &self.data.as_ref()[x2], x);
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

                                let q11 = self.data.as_ref()[x1 + (y1 * width)];
                                let q21 = self.data.as_ref()[x2 + (y1 * width)];
                                let q12 = self.data.as_ref()[x1 + (y2 * width)];
                                let q22 = self.data.as_ref()[x2 + (y2 * width)];
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

    pub fn resample(&self, scale: f32, sampling_method: SamplingMethod) -> Signal<S, Vec<S>> {
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
                                sampled_data.push(self.data.as_ref()[x as usize]);
                            } else if x >= length as f32 {
                                sampled_data.push(self.data.as_ref()[length - 1]) 
                            } else {
                                sampled_data.push(self.data.as_ref()[x as usize]);
                            }
                        }
                        SamplingMethod::Linear => {
                            let x1 = x as usize;
                            let y;
                            if x >= (self.data.as_ref().len() - 1) as f32 {
                                // linear extrapolation
                                let x0 = (x - 1.0) as usize;
                                let preceeding_val = linear_interpolate(x0 as f32, &self.data.as_ref()[x0], x1 as f32, &self.data.as_ref()[x1], x);
                                let inc = self.data.as_ref()[x1].sub(preceeding_val);
                                y = self.data.as_ref()[x1].sub(inc);
                            } else {
                                let x2 = (x + 1.0) as usize;
                                y = linear_interpolate(x1 as f32, &self.data.as_ref()[x1], x2 as f32, &self.data.as_ref()[x2], x);
                            }
                            sampled_data.push(y);
                        }
                        _ => panic!("unsupported sampling method")
                    }
                    x += step;
                }
                let new_length = sampled_data.len();
                Signal::new(SignalShape::OneDimensional(new_length), sampled_data)
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
                                let q11 = self.data.as_ref()[x1 + (y1 * width)];
                                let q21 = self.data.as_ref()[x2 + (y1 * width)];
                                let q12 = self.data.as_ref()[x1 + (y2 * width)];
                                let q22 = self.data.as_ref()[x2 + (y2 * width)];
                                let p = bilinear_interpolate(x1 as f32, y1 as f32, x2 as f32, y2 as f32, &q11, &q21, &q12, &q22, sx, sy);

                                sampled.push(p);
                            }
                        }
                    }
                    _ => panic!("unsupported sampling method")
                }

                Signal::new(SignalShape::TwoDimensional(new_width, new_height), sampled)
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
    Signal::new(SignalShape::OneDimensional(size), data)
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
