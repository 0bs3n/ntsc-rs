use crate::sample::Sample;
use image::Rgb;

pub mod sample;
pub mod crop;
pub mod signal;
pub mod ntsc;

impl Sample for u8 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }
    fn scale_f32(&self, scalar: f32) -> Self {
        (*self as f32 * scalar) as Self
    }
    fn add(&self, other: Self) -> Self {
        self.saturating_add(other)
    }
    fn sub(&self, other: Self) -> Self {
        self.saturating_sub(other)
    }
    fn depth(&self) -> usize {
        return 1;
    }
    fn zero() -> Self {
        return 0u8;
    }
}

impl Sample for f32 {
    fn to_f32(&self) -> f32 {
        *self
    }
    fn scale_f32(&self, scalar: f32) -> Self {
        *self * scalar
    }
    fn add(&self, other: Self) -> Self {
        self + other
    }
    fn sub(&self, other: Self) -> Self {
        self - other    
    }
    fn depth(&self) -> usize {
        return 1;
    }
    fn zero() -> Self {
        return 0f32;
    }
}

impl Sample for image::Luma<u8> {
    fn to_f32(&self) -> f32 {
        self.0[0] as f32
    }
    fn scale_f32(&self, scalar: f32) -> Self {
        image::Luma([(self.0[0] as f32 * scalar) as u8])
    }
    fn add(&self, other: Self) -> Self {
        image::Luma([(self.0[0].saturating_add(other.0[0]))])
    }
    fn sub(&self, other: Self) -> Self {
        image::Luma([self.0[0].saturating_sub(other.0[0])])
    }
    fn depth(&self) -> usize {
        return 1;
    }
    fn zero() -> Self {
        image::Luma([0u8])
    }
}
    
impl Sample for Rgb<u8> {
    fn to_f32(&self) -> f32 {
        self.0[0] as f32
    }
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
    fn depth(&self) -> usize {
        return 3;
    }
    fn zero() -> Self {
        Rgb([0u8, 0u8, 0u8])
    }
}

#[cfg(test)]
mod tests {
    use image::RgbImage;
    use image::{ImageReader, Rgb};
    use std::fs::File;
    use std::io::{ Read, Write };


    use crate::signal::{ 
        Signal, 
        Kernel,
        SignalShape, 
        moving_average_kernel, 
        SamplingMethod, 
        linear_interpolate, 
        bilinear_interpolate
    };
    use crate::crop::{ CropParams, CropDirection, crop_symmetric_2d, crop_symmetric_1d };
    use crate::ntsc::*;

    #[test]
    fn it_filters_image() {
        let gaussian_kernel = Kernel::new(
            SignalShape::TwoDimensional(5, 5),
            vec![ 0.0318, 0.0375, 0.0397, 0.0375, 0.0318,
                        0.0375, 0.0443, 0.0468, 0.0443, 0.0375,
                        0.0397, 0.0468, 0.0494, 0.0468, 0.0397,
                        0.0375, 0.0443, 0.0468, 0.0443, 0.0375,
                        0.0318, 0.0375, 0.0397, 0.0375, 0.0318 ]
        );

        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();

        let signal = Signal::new(
            SignalShape::TwoDimensional(img.width() as usize, img.height() as usize),
            img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>()
        );

        let filtered = signal.filter(&gaussian_kernel);

        let img = image::ImageBuffer::from_fn(
            img.width(), img.height(), |x, y| {
                let index = (y * img.width() + x) as usize;
                filtered.data[index]
            }
        );

        img.save("../filtered_2d.png").unwrap();
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
        img.save("/home/sen/Projects/RS170/crop_test.png").unwrap();
    }

    #[test]
    fn it_crops_1d() {
        let crop_width = 8;
        let mut img = RgbImage::new(10, 10);
        for (_, _, pixel) in img.enumerate_pixels_mut() {
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
        img.save("/home/sen/Projects/RS170/crop_test.png").unwrap();
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

            let signal = Signal::new(SignalShape::TwoDimensional(cropped_width, cropped_height), pixels);

            let scale = NTSC_VISIBLE_LINE_COUNT as f32 / cropped_width as f32;
            let scaled_width = (cropped_width as f32 * scale) as u32;
            let scaled_height = (cropped_height as f32 * scale) as u32;

            let mut scaled = signal.resample(scale, SamplingMethod::Bilinear);
            let kernel = crate::signal::moving_average_kernel_new(SignalShape::TwoDimensional(5, 5));
            crate::ntsc::horizontal_filter(&mut scaled, &kernel);

            let _filtered_image = image::ImageBuffer::from_fn(
                scaled_width, scaled_height, |x, y| {
                    let index = (y * scaled_width + x) as usize;
                    scaled.data[index]
                }
            );

            _filtered_image.save("/home/sen/Projects/RS170/ntsc_test.png").unwrap();
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
    fn it_filters_new_1d() {

        // this API sucks
        let kernel = crate::signal::moving_average_kernel_new(SignalShape::OneDimensional(3));

        let data: Vec<u8> = vec![10, 5, 10, 5, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0];
        println!("original dat:\t{:02x?}", data);

        let shape = SignalShape::OneDimensional(data.len());
        let signal = Signal::new(shape, data);
        let filtered_data = signal.filter(&kernel);
        println!("filtered new:\t{:02x?}", filtered_data.data);

        let data: Vec<Rgb<u8>> = vec![Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255]), Rgb([0,0,0]), 
                                      Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255]), Rgb([0,0,0]), 
                                      Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255])];
        println!("original dat:\t{:02x?}", data);

        let shape = SignalShape::OneDimensional(data.len());
        let signal = Signal::new(shape, data);
        let filtered_data = signal.filter(&kernel).data;

        println!("filtered new:\t{:02x?}", filtered_data);
    }

    #[test]
    fn it_filters_new_2d() {
        let kernel = crate::signal::moving_average_kernel_new(SignalShape::TwoDimensional(3, 3));

        let data = Signal::<u8, Vec<u8>>::new(
            SignalShape::TwoDimensional(7, 7),
            vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
                 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22,
                 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
                 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30]
        );
        println!("original:\n{}", data);
        let filtered: Signal<u8, Vec<u8>> = data.filter(&kernel);
        println!("filtered:\n{}", filtered);

        let data = Signal::new(
            SignalShape::TwoDimensional(7, 5),
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        println!("original:\n{}", data);
        let filtered: Signal<u8, Vec<u8>> = data.filter(&kernel);
        println!("filtered:\n{}", filtered);

        let data = Signal::new(
            SignalShape::TwoDimensional(6, 6),
            vec![
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            ]
        );
        println!("original:\n{}", data);
        let filtered = data.filter(&kernel);
        println!("filtered:\n{}", filtered);
    }

    #[test]
    fn it_filters_horizontal_image() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let mut img = img_.to_rgb8();

        let height = img.height() as usize;
        let width = img.width() as usize;

        let pixels = img.as_mut();

        let kernel = Kernel::new(
            SignalShape::OneDimensional(10),
            moving_average_kernel(10)
        );

        for i in 0..height {
            let row_start =  i as usize * width * 3;
            let row_end = row_start + (width * 3);

            let row_bytes = &mut pixels[row_start..row_end];
            let mut row_pixels: &mut [Rgb<u8>] = unsafe {
                std::slice::from_raw_parts_mut(row_bytes.as_mut_ptr() as *mut Rgb<u8>, width)
            };

            let row_pixel_len = row_pixels.len();

            Signal::new(
                SignalShape::OneDimensional(row_pixel_len),
                &mut row_pixels,
            ).filter_in_place(&kernel);
        }

        img.save("/home/sen/Projects/RS170/horizontal_moving_avg.png").unwrap();
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

        let signal = Signal::new(
            SignalShape::TwoDimensional(img.width() as usize, img.height() as usize),
            pixels
        );

        let sampled = signal.resample(2.0, SamplingMethod::Bilinear);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                sampled.data[index]
            }
        );
        sampled_image.save("/home/sen/Projects/RS170/double.png").unwrap();

        let scale = 0.5;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let sampled = signal.resample(scale, SamplingMethod::Bilinear);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                sampled.data[index]
            }
        );
        sampled_image.save("/home/sen/Projects/RS170/half.png").unwrap();
    }

    #[test]
    fn it_resamples_3d_luma() {
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_luma8();

        img.save("/home/sen/Projects/RS170/luma_orig.png").unwrap();
        let scale = 2.0;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<image::Luma<u8>>>();

        let signal = Signal::new(
            SignalShape::TwoDimensional(img.width() as usize, img.height() as usize),
            pixels
        );

        let sampled = signal.resample(2.0, SamplingMethod::Bilinear);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                sampled.data[index]
            }
        );

        sampled_image.save("/home/sen/Projects/RS170/luma_resample.png").unwrap();
    }

    #[test]
    fn it_resamples_2d() {
        let shape = SignalShape::OneDimensional(10);
        let data: [u8; 10] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
        let signal = Signal::new(shape, data.to_vec());
        let sampled = signal.resample(0.5, SamplingMethod::Point);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);
        // assert_eq!(sampled, vec![0, 10, 30, 50, 70, 90]);

        let sampled = signal.resample(0.5, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);
        // assert_eq!(sampled, vec![0, 5, 25, 45, 65, 85]);

        let sampled = signal.resample(2.0, SamplingMethod::Point);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        // failing
        let sampled = signal.resample(2.0, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let signal = Signal::new(
            SignalShape::OneDimensional(7),
            vec![1, 3, 5, 7, 9, 11, 13]
        );

        let sampled = signal.resample(2.0, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let signal = Signal::new(
            SignalShape::OneDimensional(7),
            vec![1, 1, 90, 2, 100, 22, 13, 1]
        );

        let sampled = signal.resample(2.0, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let signal = Signal::new(
            SignalShape::OneDimensional(7),
            vec![1, 1, 90, 2, 100, 22, 13, 1]
        );

        let sampled = signal.resample(0.732, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let mut f = File::open("/home/sen/Projects/RS170/sine_wave_u8.bin").unwrap();
        let mut data = Vec::<u8>::new();
        f.read_to_end(&mut data).unwrap();

        let signal = Signal::new(
            SignalShape::OneDimensional(data.len()),
            &mut data
        );

        let sampled = signal.resample(0.5, SamplingMethod::Linear);
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_downscaled_0.5.bin").unwrap();
        f2.write_all(sampled.data.as_slice()).unwrap();

        let signal = Signal::new(
            SignalShape::OneDimensional(data.len()),
            &mut data
        );
        let sampled = signal.resample(5.0, SamplingMethod::Linear);
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_upscaled_5.bin").unwrap();
        f2.write_all(sampled.data.as_slice()).unwrap();
    }

    #[test]
    fn it_resamples_3d() {
        let data = Signal::new(
            SignalShape::TwoDimensional(7, 7),
            vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
                 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22,
                 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
                 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30],
        );

        println!("original:\n{}", data);
        let sampled = data.resample(0.5, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);
        let sampled = data.resample(2.0, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);

        let data = Signal::new(
            SignalShape::TwoDimensional(7, 5),
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        println!("original:\n{}", data);
        let sampled = data.resample(0.5, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);
        let sampled = data.resample(2.0, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);

        let data = Signal::new(
            SignalShape::TwoDimensional(6, 6),
            vec![
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            ]
        );
        println!("original:\n{}", data);
        let sampled = data.resample(0.5, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);
        let sampled = data.resample(2.0, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);
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
