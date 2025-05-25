use crate::sample::Sample;
use std::fmt;

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

pub struct Kernel {
    pub shape: SignalShape,
    pub data: Vec<f32>
}

impl Kernel {
    pub fn moving_average(shape: SignalShape) -> Kernel {
        match shape { 
            SignalShape::OneDimensional(size) => 
                Kernel { data: vec![1.0 / size as f32; size], shape },
            SignalShape::TwoDimensional(width, height) => 
                Kernel { data: vec![1.0 / (width * height) as f32; width * height], shape }
        }
    }
}

// TODO: maybe make Signal and Kernel implementors of some trait
#[derive(Debug)]
pub struct Signal<S: Sample> {
    pub shape: SignalShape,
    pub data: Vec<S>
}

impl<'a, S: Sample> fmt::Display for Signal<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape {
            SignalShape::OneDimensional(_) => write!(f, "{:02x?}", self.data),
            SignalShape::TwoDimensional(width, height) => {
                for y in 0..height {
                    write!(f, "{:02x?}\n", &self.data[(y * width)..(y * width + width)])?;
                }
                Ok(())
            }
        }
    }
}

impl<S: Sample> Signal<S> {
    fn window(&self, center: usize, shape: SignalShape) -> Signal<S> {
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
        Signal {
            shape,
            data: output
        }
    }

    pub fn filter(&self, kernel: &Kernel) -> Signal<S> {
        let mut output = Vec::<S>::new();
        for i in 0..self.data.len() {
            let sample = S::convolve(
                self.window(i, kernel.shape).data.as_slice(), 
                kernel.data.as_slice());
            output.push(sample);
        }
        Signal { shape: self.shape, data: output }
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
                                sampled_data.push(self.data[x.floor() as usize]);
                            }
                        }
                        SamplingMethod::Linear => {
                            let x1 = x.floor() as usize;
                            let y;
                            if x >= (self.data.len() - 1) as f32 {
                                // linear extrapolation
                                let x0 = (x - 1.0).floor() as usize;
                                let preceeding_val = linear_interpolate(x0 as f32, &self.data[x0], x1 as f32, &self.data[x1], x);
                                let inc = self.data[x1].sub(preceeding_val);
                                y = self.data[x1].sub(inc);
                            } else {
                                let x2 = (x + 1.0).floor() as usize;
                                y = linear_interpolate(x1 as f32, &self.data[x1], x2 as f32, &self.data[x2], x);
                            }
                            sampled_data.push(y);
                        }
                        _ => panic!("unsupported sampling method")
                    }
                    x += step;
                }
                let new_length = sampled_data.len();
                Signal { data: sampled_data, shape: SignalShape::OneDimensional(new_length) }
            }
            SignalShape::TwoDimensional(width, height) => {
                let new_width  = (width as f32 * scale) as usize;
                let new_height = (height as f32 * scale) as usize;
                let mut sampled = Vec::<S>::with_capacity(new_width * new_height);
                match sampling_method {
                    SamplingMethod::Bilinear => {
                        for dy in 0..new_height {
                            let sy = (dy as f32 / (new_height - 1) as f32 ) * (height - 1) as f32;
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

                Signal { data: sampled, shape: SignalShape::TwoDimensional(new_width, new_height) }
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

pub fn moving_average_kernel_new(radius: usize) -> Kernel {
    let size = radius * 2 + 1;
    let data = vec![1.0 / (radius * 2 + 1) as f32; radius * 2 + 1];
    Kernel {
        data,
        shape: SignalShape::OneDimensional(size)
    }
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
    (x.floor() as usize, x.ceil() as usize, y.floor() as usize, y.ceil() as usize)
}
