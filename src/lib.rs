use image::{ImageReader, Rgb, Luma};
use std::ops::{ Mul, Add };
use std::fmt::Debug;


use std::fs::File;
use std::io::{ Read, Write };

const NTSC_VISIBLE_LINE_COUNT: usize = 480;

// Y′=0.299⋅R+0.587⋅G+0.114⋅B
// weights above sum to 1, resulting Y' will always be <= 255
// if input r, g, and b are
fn rgb8_to_luma(r: u8, g: u8, b: u8) -> u8 {
    (((r as usize * 299) + (g as usize * 587) + (b as usize * 114)) / 1000) as u8
}

fn linear_interpolate<D: Signal>(x1: f32, y1: &D, x2: f32, y2: &D, x: f32) -> D 
{
    if x1 == x2 && x == x1 { 
        return *y1;
    }
    let dist = (x2 - x1).abs();
    let x1_weight = x2 - x;
    let x2_weight = dist - x1_weight;

    y1.scale_f32(x1_weight).add(y2.scale_f32(x2_weight))
}

fn bilinear_interpolate<D: Signal>(x1: f32, y1: f32, x2: f32, y2: f32, q11: &D, q21: &D, q12: &D, q22: &D, x: f32, y: f32) -> D 
{
    let r1 = linear_interpolate(x1, q11, x2, q21, x);
    let r2 = linear_interpolate(x1, q12, x2, q22, x);
    linear_interpolate(y1, &r1, y2, &r2, y)
}

fn crop2d(i: image::DynamicImage, w: usize, h: usize) {
    // find the horizontal center
    // capture window of w pixels offset from the left by (i.width() - w) / 2
}

pub fn dump_image(filename: &str) {
    let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
    let img = img_.to_rgb8();
    let mut luma_img: image::GrayImage = image::GrayImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        println!("{:?}, {:?}, {:?}", x, y, pixel);
        println!("Luma for RBG({}, {}, {}): {}", 
            pixel.0[0], pixel.0[1], pixel.0[2], 
            rgb8_to_luma(pixel.0[0], pixel.0[1], pixel.0[2]));
        luma_img.put_pixel(x, y, image::Luma([rgb8_to_luma(pixel.0[0], pixel.0[1], pixel.0[2])]));
    }
    let _ = luma_img.save("/home/sen/Projects/RS170/test_pattern_luma8_mine.png");
    let _ = img_.to_luma8().save("/home/sen/Projects/RS170/test_pattern_luma8.png");
}

enum SamplingMethod {
    Point,
    Linear,
    Bilinear
}

fn sample2D<D: Signal>(data: &[D], scale: f32, sampling_method: SamplingMethod, center_grid: bool) -> Vec<D> {
    // TODO: is this correct? Should we leave it at 0?
    let mut x = if center_grid { 0.5 } else { 0.0 };
    let mut sampled_data = Vec::<D>::new();
    let step = 1.0 / scale;
    // println!("step: {}", step);

    while x < data.len() as f32 {
        // println!("x: {}", x);
        match sampling_method {
            SamplingMethod::Point => {
                if x == 0.0 {
                    sampled_data.push(data[x as usize]);
                } else if x >= data.len() as f32 {
                    sampled_data.push(data[data.len() - 1]) 
                } else {
                    sampled_data.push(data[x.floor() as usize]);
                }
            }
            SamplingMethod::Linear => {
                let x1 = x.floor() as usize;
                let mut y;
                if x >= (data.len() - 1) as f32 {
                    println!("x = {}, data len: {}", x, (data.len() - 1) as f32);

                    // linear extrapolation
                    let x0 = (x - 1.0).floor() as usize;
                    println!("data[{}]: {:?}, data[{}]: {:?}", x0, data[x0], x1, data[x1]);
                    println!("x0: {}, x1: {}", x0, x1);

                    let preceeding_val = linear_interpolate(x0 as f32, &data[x0], x1 as f32, &data[x1], x);

                    println!("preceeding_val: {:?}", preceeding_val);

                    let inc = data[x1].sub(preceeding_val);

                    println!("inc: {:?}", inc);

                    y = data[x1].sub(inc);

                    println!("y = {:?}", y);
                } else {
                    let x2 = (x + 1.0).floor() as usize;
                    println!("data[{}]: {:?}, data[{}]: {:?}", x1, data[x1], x2, data[x2]);
                    y = linear_interpolate(x1 as f32, &data[x1], x2 as f32, &data[x2], x);
                    println!("y = {:?}", y);
                }
                sampled_data.push(y);
            }
            _ => panic!("unsupported sampling method")
        }
        x += step;
    }
    sampled_data
}

fn bilinear_interpolate_get_neighbors(x: f32, y: f32) -> (usize, usize, usize, usize) {
    (x.floor() as usize, x.ceil() as usize, y.floor() as usize, y.ceil() as usize)
}

fn sample3D<D: Signal>(data: &[D], width: usize, scale: f32, method: SamplingMethod) -> Vec<D> 
{
    let orig_len = data.len();
    assert!(orig_len % width == 0);
    
    let height = orig_len / width;

    let new_width  = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;
    let mut sampled = Vec::<D>::with_capacity(new_width * new_height);

    // find a better way?
    // unsafe { sampled.set_len(new_width * new_height) }

    for dy in 0..new_height {
        let sy = (dy as f32 / (new_height - 1) as f32 ) * (height - 1) as f32;
        if sy.ceil() >= height as f32 { break }

        for dx in 0..new_width {
            let sx = (dx as f32 / (new_width - 1) as f32 ) * (width - 1) as f32;
            let (x1, x2, y1, y2) = bilinear_interpolate_get_neighbors(sx, sy);

            let q11 = data[x1 + (y1 * width)];
            let q21 = data[x2 + (y1 * width)];
            let q12 = data[x1 + (y2 * width)];
            let q22 = data[x2 + (y2 * width)];
            let p = bilinear_interpolate(x1 as f32, y1 as f32, x2 as f32, y2 as f32, &q11, &q21, &q12, &q22, sx, sy);

            // sampled[(dy * new_width) + dx] = p;
            sampled.push(p);
        }
    }

    sampled
}



pub trait Signal: Debug + Copy {
    fn scale_f32(&self, scalar: f32) -> Self;
    fn add(&self, other: Self) -> Self;
    fn sub(&self, other: Self) -> Self;
    // TODO: if the scale_f32 function truncates the resulting value, the subsequent adds lose
    // at worst 0.99 units of preceision per element in the window. if they round instead, worst loss
    // is 0.5 units. with for convolutions with large window sizes, this causes distortions in the 
    // output signal, even if every element in the window is the same.
    //
    // i.e., convolution over &[u8], window contains all 255. kernel radius is 10.
    // scale_f32 multiplies the value 255 by the weight (1.0 / (10 * 2 + 1)) == 0.04761905
    // 255 * 0.04761905 == 12.142857750000001, which scale_f32 truncates before returning
    // to accomodate the Self return type requirement, to 12. After kernel scaling has been
    // applied, the intermediate array is [12u8; 21].
    // this is then summed: 12 * 21 == 252, i.e. not 255.
    //
    // This leads to an overall slight attenuation of the signal due to precision loss
    // if instead scale_by rounded to the nearest rather than truncating, precision loss would be limited
    // to 0.5 units - and the result might be a slight attenuation or amplification of the signal,
    // but a smaller one. However calling `round()` in scale_f32 has a massive performance hit,
    // so sticking with greater precision loss but better perf. 
    // worst case for truncation: roughly one unit of loss per window element, so
    // worst case is an attenuation of kernel_size to every sample. pretty bad with 
    // large window sizes.
    fn convolve(window: &[Self], kernel: &[f32]) -> Self {
        window.iter().zip(kernel)
        .map(|(&w, &k)| w.scale_f32(k))
        .reduce(|a, b| a.add(b)).unwrap()
    }
    fn filter<D: Signal>(data: &[D], kernel: &[f32]) -> Vec<D> {
        assert!(kernel.len() < data.len());
        let mut output = Vec::<D>::new();

        let kernel_size = kernel.len();
        let r = (kernel_size - 1) / 2;

        let left_padding = vec![data[0]; r];
        let right_padding = vec![data[data.len() - 1]; r];

        for i in 0..data.len() {
            let window = if i < r {
                &[&left_padding[0..r - i], &data[i..i + r + 1]].concat()
            } else if i > (data.len() - r - 1) {
                &[&data[i - r..data.len()], &right_padding[0..r - ((data.len() - 1) - i)]].concat()
            } else {
                &data[i - r ..= i + r]
            };

            output.push(D::convolve(window, kernel));
        }
        output
    }
}

impl Signal for u8 {
    fn scale_f32(&self, scalar: f32) -> Self {
        (*self as f32 * scalar) as Self
    }
    fn add(&self, other: Self) -> Self {
        self + other
    }
    fn sub(&self, other: Self) -> Self {
        self - other    
    }
}

impl Signal for f32 {
    fn scale_f32(&self, scalar: f32) -> Self {
        *self * scalar
    }
    fn add(&self, other: Self) -> Self {
        self + other
    }
    fn sub(&self, other: Self) -> Self {
        self - other    
    }
}

impl Signal for image::Luma<u8> {
    fn scale_f32(&self, scalar: f32) -> Self {
        image::Luma([(self.0[0] as f32 * scalar) as u8])
    }
    fn add(&self, other: Self) -> Self {
        image::Luma([(self.0[0].saturating_add(other.0[0]))])
    }
    fn sub(&self, other: Self) -> Self {
        image::Luma([self.0[0].saturating_sub(other.0[0])])
    }
}
    
impl Signal for Rgb<u8> {
    fn scale_f32(&self, scalar: f32) -> Self {
        let r = (self.0[0] as f32 * scalar) as u8;
        let g = (self.0[1] as f32 * scalar) as u8;
        let b = (self.0[2] as f32 * scalar) as u8;
        Rgb([r, g, b])
    }
    fn add(&self, other: Self) -> Self {
        let r = self.0[0].saturating_add(other.0[0]);
        let g = self.0[1].saturating_add(other.0[1]);
        let b = self.0[2].saturating_add(other.0[2]);
        Rgb([r, g, b])
    }
    fn sub(&self, other: Self) -> Self {
        let r = self.0[0].saturating_sub(other.0[0]);
        let g = self.0[1].saturating_sub(other.0[1]);
        let b = self.0[2].saturating_sub(other.0[2]);
        Rgb([r, g, b])
    }
}

fn crop_symmetric_1d<D>(data: &[D], width: usize) -> &[D] {
    let offset = (data.len() - width) / 2;
    &data[offset..data.len() - offset]
}

fn crop_symmetric_width_2d<D: Clone>(data: &[D], width: usize, height: usize, crop_to: usize) -> (usize, usize, Vec<D>) {
    let mut cropped = Vec::<D>::new();
    for i in 0..height {
        let row_start =  i as usize * width as usize;
        let row_end = row_start + width as usize;
        let row = &data[row_start..row_end];
        cropped.append(&mut crop_symmetric_1d(row, crop_to).to_vec());
    }
    (crop_to, height, cropped)
}

#[derive(Debug)]
enum CropDirection {
    Vertical,
    Horizontal
}
struct CropParams {
    magnitude: usize,
    direction: CropDirection
}

fn crop_symmetric_2d<D: Clone>(data: &[D], width: usize, height: usize, crop_params: CropParams) -> (usize, usize, Vec<D>) {
    let mut cropped = Vec::<D>::new();

    for y in 0..height {
        let row_start =  y as usize * width as usize;
        let row_end = row_start + width as usize;
        let row = &data[row_start..row_end];

        match crop_params.direction {
            CropDirection::Vertical   => {
                let offset = (height - crop_params.magnitude) / 2;
                if y < offset { continue }
                if y >= height - offset { 
                    return (width, crop_params.magnitude, cropped) 
                }
                println!("offset: {}, y: {}, row_start: {}, row_end: {}", offset, y, row_start, row_end);
                println!("row length: {}", row.len());
                cropped.append(&mut row.to_vec());
            }
            CropDirection::Horizontal => {
                let offset = (width - crop_params.magnitude) / 2;
                cropped.append(&mut row[offset..row.len() - offset - 1].to_vec());
            }
        }
    }
    return (crop_params.magnitude, height, cropped);
}

fn crop_for_ntsc<D: Clone>(data: &[D], width: usize, height: usize) -> (usize, usize, Vec<D>) {
    // let output = Vec::<D>::new();
    let cp: CropParams;
    let current_ar = height as f32 / width as f32;
    if current_ar < 0.75 { // Aspect ratio greater than 4:3, i.e. 16:9
        let magnitude = (height as f32 / 0.75) as usize;
        cp = CropParams {
            magnitude,
            direction: CropDirection::Horizontal
        };
    } else { // aspect ratio less than 4:3, i.e. 1:1
        let magnitude = (width as f32 * 0.75) as usize;
        cp = CropParams {
            magnitude,
            direction: CropDirection::Vertical
        };
    }
    crop_symmetric_2d(data, width, height, cp)
}

fn moving_average_kernel(radius: usize) -> Vec<f32> {
    vec![1.0 / (radius * 2 + 1) as f32; radius * 2 + 1]
}

fn ntsc_process_frame<D: image::Pixel + Signal>(pixels: &[D], width: usize, height: usize) -> (usize, usize, Vec<D>) {
    let (cropped_width, cropped_height, pixels) = 
        crop_for_ntsc(&pixels, width as usize, height as usize);


    let scale = NTSC_VISIBLE_LINE_COUNT as f32 / cropped_height as f32;
    let scaled_width = (cropped_width as f32 * scale) as u32;
    let scaled_height = (cropped_height as f32 * scale) as u32;


    let mut sampled = sample3D(&pixels, cropped_width as usize, scale, SamplingMethod::Bilinear);

    let kernel = moving_average_kernel(5);

    for y in 0..scaled_height {
        let start =  y as usize * scaled_width as usize;
        let end = start + scaled_width as usize;
        let row = &mut sampled[start..end];
        let lpf_row = Rgb::filter(row, &kernel);
        for (dst, src) in row.iter_mut().zip(lpf_row.iter()) {
            *dst = *src;
        }
    }
    (scaled_width as usize, scaled_height as usize, sampled)
}

#[cfg(test)]
mod tests {
    use image::{imageops::crop, ImageBuffer, RgbImage};

    use super::*;

    #[test]
    fn it_dumps() {
        dump_image("/home/sen/Projects/RS170/test_pattern.png");
    }

    #[test]
    fn it_crops_for_ntsc() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();
        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>();
        let (cropped_width, cropped_height, new_pixels) = 
            crop_for_ntsc(&pixels, img.width() as usize, img.height() as usize);
        let img = image::ImageBuffer::from_fn(
            cropped_width as u32, cropped_height as u32, |x, y| {
                let index = (y * cropped_width as u32 + x) as usize;
                new_pixels[index]
            }
        );
        img.save("cropped_for_ntsc.png").unwrap();
    }

    #[test]
    fn it_crops_2d() {
        let crop_width = 400;
        let crop_params = CropParams { 
            magnitude: crop_width, 
            direction: CropDirection::Vertical 
        };
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();
        let pixels = img.enumerate_pixels()
            .map(|(_, _, pixel)| *pixel)
            .collect::<Vec<Rgb<u8>>>();

        let (cropped_width, cropped_height, new_pixels) = 
            crop_symmetric_2d(
                &pixels, 
                img.width() as usize, 
                img.height() as usize, 
                crop_params
            );

        println!("length: {:?}", new_pixels.len());

        let img = image::ImageBuffer::from_fn(
            cropped_width as u32, cropped_height as u32, |x, y| {
                let index = (y * cropped_width as u32 + x) as usize;
                new_pixels[index]
            }
        );
        img.save("/home/sen/Projects/RS170/crop_test.png");
    }

    #[test]
    fn it_crops_1d() {
        let crop_width = 8;
        let mut img = RgbImage::new(10, 10);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }
        let pixels = img.enumerate_pixels()
            .map(|(_, _, pixel)| *pixel)
            .collect::<Vec<Rgb<u8>>>();

        let mut new_pixels = Vec::<Rgb<u8>>::new();
        for i in 0..img.height() {
            let row_start =  i as usize * img.width() as usize;
            let row_end = row_start + img.width() as usize;
            let row = &pixels[row_start..row_end];
            new_pixels.append(&mut crop_symmetric_1d(row, crop_width).to_vec());
        }
        let img = image::ImageBuffer::from_fn(
            crop_width as u32, img.height(), |x, y| {
                let index = (y * crop_width as u32 + x) as usize;
                new_pixels[index]
            }
        );
        img.save("/home/sen/Projects/RS170/crop_test.png");
    }

    #[test]
    fn it_processes_30_frames() {
        const NTSC_VISIBLE_LINE_COUNT: usize = 640;
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();
        for _ in 0..30 {
            let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>();

            let (cropped_width, cropped_height, pixels) = 
                crop_for_ntsc(&pixels, img.width() as usize, img.height() as usize);

            let scale = NTSC_VISIBLE_LINE_COUNT as f32 / cropped_width as f32;
            let scaled_width = (cropped_width as f32 * scale) as u32;
            let scaled_height = (cropped_height as f32 * scale) as u32;


            let mut sampled = sample3D(&pixels, cropped_width as usize, scale, SamplingMethod::Bilinear);

            let kernel = moving_average_kernel(5);
            for y in 0..scaled_height {
                let start =  y as usize * scaled_width as usize;
                let end = start + scaled_width as usize;
                let row = &mut sampled[start..end];
                let lpf_row = Rgb::filter(row, &kernel);
                for (dst, src) in row.iter_mut().zip(lpf_row.iter()) {
                    *dst = *src;
                }
            }
        }
    }

    #[test]
    fn it_preprocesses_for_ntsc() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();

        // TODO: see if going back to the strategy using as_mut/unsafe is faster
        let pixels = img.enumerate_pixels()
            .map(|(_, _, pixel)| *pixel)
            .collect::<Vec<Rgb<u8>>>();

        let (pwidth, pheight, pdata) = 
            ntsc_process_frame(&pixels, img.width() as usize, img.height() as usize);

        let img = image::ImageBuffer::from_fn(
            pwidth as u32, pheight as u32, |x, y| {
                let index = (y * pwidth as u32 + x) as usize;
                pdata[index]
            }
        );

        img.save("/home/sen/Projects/RS170/scaled_and_filtered_and_cropped.png").unwrap();
    }

    #[test]
    fn it_filters_2d() {
        let radius = 10;
        let kernel = moving_average_kernel(radius);

        /*
        let data: Vec<u8> = vec![10, 5, 10, 5, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0];
        let filtered_data = u8::filter2d(data.as_slice(), &kernel);
        println!("original:\t{:02x?}", data);
        println!("filtered:\t{:02x?}", filtered_data);
        let filtered_data = u8::moving_average(data.as_slice(), radius);
        println!("filtered2:\t{:02x?}", filtered_data);

        let data: Vec<Rgb<u8>> = vec![Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255]), Rgb([0,0,0]), 
                                      Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255]), Rgb([0,0,0]), 
                                      Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255])];

        let filtered_data = u8::filter2d(data.as_slice(), &kernel);
        println!("original:\t{:02x?}", data);
        println!("filtered:\t{:02x?}", filtered_data);
        let filtered_data = u8::moving_average(data.as_slice(), radius);
        println!("filtered2:\t{:02x?}", filtered_data);
        */

        let data: Vec<u8> = vec![
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        let filtered_data = u8::filter(data.as_slice(), &kernel);
        println!("original:\t{:02x?}", data);
        println!("filtered:\t{:02x?}", filtered_data);

        let data: Vec<Rgb<u8>> = vec![
            Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
        ];
        let filtered_data = u8::filter(data.as_slice(), &kernel);
        println!("original:\t{:02x?}", data);
        println!("filtered:\t{:02x?}", filtered_data);
    }

    #[test]
    fn it_filters_horizontal_image() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let mut img = img_.to_rgb8();

        let height = img.height() as usize;
        let width = img.width() as usize;

        let pixels = img.as_mut();

        let kernel = moving_average_kernel(10);

        for i in 0..height {
            let row_start =  i as usize * width * 3;
            let row_end = row_start + (width * 3);

            let row_bytes = &mut pixels[row_start..row_end];
            let row_pixels: &mut [Rgb<u8>] = unsafe {
                std::slice::from_raw_parts_mut(row_bytes.as_mut_ptr() as *mut Rgb<u8>, width)
            };

            let lpf_row = Rgb::filter(row_pixels, &kernel);

            // println!("{:?}", row_pixels);
            // println!("{:?}", lpf_row);

            for (dst, src) in row_pixels.iter_mut().zip(lpf_row.iter()) {
                *dst = *src;
            }
        }

        img.save("/home/sen/Projects/RS170/horizontal_moving_avg.png");
    }

    #[test]
    fn it_resamples_3d_rgb() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();

        img.save("/home/sen/Projects/RS170/orig.png").unwrap();


        let scale = 2.0;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>();
        let sampled = sample3D(&pixels, img.width() as usize, scale, SamplingMethod::Bilinear);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                sampled[index]
            }
        );
        sampled_image.save("/home/sen/Projects/RS170/double.png").unwrap();

        let scale = 0.5;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>();
        let sampled = sample3D(&pixels, img.width() as usize, scale, SamplingMethod::Bilinear);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                sampled[index]
            }
        );
        sampled_image.save("/home/sen/Projects/RS170/half.png").unwrap();
    }

    #[test]
    fn it_resamples_3d_luma() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_luma8();

        img.save("/home/sen/Projects/RS170/orig.png").unwrap();
        let scale = 2.0;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<image::Luma<u8>>>();
        let sampled = sample3D(&pixels, img.width() as usize, 2.0, SamplingMethod::Bilinear);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                sampled[index]
            }
        );

        sampled_image.save("/home/sen/Projects/RS170/new.png").unwrap();
    }

    #[test]
    fn it_resamples_2d() {
        let data: [u8; 10] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
        println!("{:?}", data);
        let sampled = sample2D(&data, 0.5, SamplingMethod::Point, false);
        println!("{:?}", sampled);
        // assert_eq!(sampled, vec![0, 10, 30, 50, 70, 90]);

        let sampled = sample2D(&data, 0.5, SamplingMethod::Linear, true);
        println!("{:?}", sampled);
        // assert_eq!(sampled, vec![0, 5, 25, 45, 65, 85]);

        let sampled = sample2D(&data, 2.0, SamplingMethod::Point, false);
        println!("{:?}", sampled);

        let sampled = sample2D(&data, 2.0, SamplingMethod::Linear, false);
        println!("{:?}", sampled);

        let sampled = sample2D(&[1, 3, 5, 7, 9, 11, 13], 2.0, SamplingMethod::Linear, false);
        println!("{:?}", sampled);

        let sampled = sample2D(&[1, 1, 90, 2, 100, 22, 13, 1], 2.0, SamplingMethod::Linear, false);
        println!("{:?}", sampled);

        let sampled = sample2D(&[1, 1, 90, 2, 100, 22, 13, 1], 0.732, SamplingMethod::Linear, true);
        println!("{:?}", sampled);

        let mut f = File::open("/home/sen/Projects/RS170/sine_wave.bin").unwrap();
        let mut data = Vec::<u8>::new();
        f.read_to_end(&mut data).unwrap();
        let sampled = sample2D(&data, 0.5, SamplingMethod::Linear, true);
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_downscaled_0.5.bin").unwrap();
        f2.write_all(sampled.as_slice()).unwrap();

        let sampled = sample2D(&data, 5.0, SamplingMethod::Linear, false);
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_upscaled_5.bin").unwrap();
        f2.write_all(sampled.as_slice()).unwrap();
    }

    struct Point {
        x: f32,
        y: f32
    }

    #[test]
    fn it_interpolates_linear() {
        let p0 = Point { x: 1.0, y: 15.0 };
        let p1 = Point { x: 2.0, y: 25.0 };
        assert_eq!(linear_interpolate(p0.x, &p0.y, p1.x, &p1.y, 1.50), 20.0);
        assert_eq!(linear_interpolate(p0.x, &p0.y, p1.x, &p1.y, 1.25), 17.5);
        assert_eq!(linear_interpolate(p0.x, &p0.y, p1.x, &p1.y, 1.75), 22.5);
        assert_eq!(linear_interpolate(p0.x, &p0.y, p1.x, &p1.y, 1.01), 15.1);
        assert_eq!(linear_interpolate(p0.x, &p0.y, p1.x, &p1.y, 1.99), 24.9);
        // figure out how much precision is necessary. This works but fails due to
        // precision error
        // assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.33333), 18.33333);
    }

    #[test]
    fn it_interpolates_bilinear() {
        assert_eq!(bilinear_interpolate(0.0, 0.0, 1.0, 1.0, &100.0, &200.0, &300.0, &100.0, 0.5, 0.5), 175.0);
        assert_eq!(bilinear_interpolate(0.0, 0.0, 1.0, 1.0, &100.0, &200.0, &300.0, &100.0, 0.2, 0.8), 232.0);
        assert_eq!(bilinear_interpolate(0.0, 0.0, 1.0, 1.0, &100.0, &200.0, &300.0, &100.0, 0.231, 0.587), 199.8209);

        println!("{}", bilinear_interpolate(1.0, 0.0, 2.0, 1.0, &255.0, &0.0, &76.0, &255.0, 1.0, 1.5));

    }
}
