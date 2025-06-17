use crate::sample::Sample;
use image::Rgb;

pub mod sample;
pub mod crop;
pub mod signal;
pub mod ntsc;

impl Sample for u8 {
    fn flatten_and_push(&self, out: &mut Vec<f32>) {
        out.push(*self as f32) 
    }
    fn from_signal_slice(s: &[f32]) -> Self {
        return s[0] as u8;
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
    fn depth() -> usize {
        return 1;
    }
    fn zero() -> Self {
        return 0u8;
    }
}

impl Sample for f32 {
    fn flatten_and_push(&self, out: &mut Vec<f32>) {
        out.push(*self) 
    }
    fn from_signal_slice(s: &[f32]) -> Self {
        s[0]
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
    fn depth() -> usize {
        return 1;
    }
    fn zero() -> Self {
        return 0f32;
    }
}

impl Sample for image::Luma<u8> {
    fn flatten_and_push(&self, out: &mut Vec<f32>) {
        out.push(self.0[0] as f32);
    }
    fn from_signal_slice(s: &[f32]) -> Self {
        image::Luma([s[0] as u8])
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
    fn depth() -> usize {
        return 1;
    }
    fn zero() -> Self {
        image::Luma([0u8])
    }
}
    
impl Sample for Rgb<u8> {
    fn flatten_and_push(&self, out: &mut Vec<f32>) {
        out.push(self.0[0] as f32);
        out.push(self.0[1] as f32);
        out.push(self.0[2] as f32);
    }
    fn from_signal_slice(s: &[f32]) -> Self {
        let r = s[0] as u8;
        let g = s[1] as u8;
        let b = s[2] as u8;
        Rgb([r, g, b])
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
    fn depth() -> usize {
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
            1,
            vec![ 0.0318, 0.0375, 0.0397, 0.0375, 0.0318,
                        0.0375, 0.0443, 0.0468, 0.0443, 0.0375,
                        0.0397, 0.0468, 0.0494, 0.0468, 0.0397,
                        0.0375, 0.0443, 0.0468, 0.0443, 0.0375,
                        0.0318, 0.0375, 0.0397, 0.0375, 0.0318 ]
        );

        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_rgb8();

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(img.width() as usize, img.height() as usize),
            img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>().as_slice()
        );

        use crate::sample::Sample;
        let filtered = signal.filter(&gaussian_kernel);
        let final_pixels = Rgb::<u8>::unflatten(&filtered);

        let img = image::ImageBuffer::from_fn(
            img.width(), img.height(), |x, y| {
                let index = (y * img.width() + x) as usize;
                final_pixels[index]
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
        let img = img_.to_luma8();
        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<image::Luma<u8>>>();

        for _ in 0..30 {
            let (cropped_width, cropped_height, mut pixels) = 
                crop_for_ntsc(&pixels, img.width() as usize, img.height() as usize);

            let signal = Signal::<Vec<f32>>::from_samples(
                SignalShape::TwoDimensional(cropped_width, cropped_height), 
                &mut pixels);
                
            // let signal = Signal::<Vec<f32>>::new(
                // SignalShape::TwoDimensional(cropped_width, cropped_height), 
                // 1, 
                // pixels.into_iter().map(|x| x.0[0] as f32).collect());

            let scale = NTSC_VISIBLE_LINE_COUNT as f32 / cropped_width as f32;
            // let scaled_width = (cropped_width as f32 * scale) as u32;
            // let scaled_height = (cropped_height as f32 * scale) as u32;

            let mut scaled = signal.resample(scale, SamplingMethod::Bilinear);
            let kernel = crate::signal::moving_average_kernel_new(SignalShape::TwoDimensional(5, 5));
            crate::ntsc::horizontal_filter(&mut scaled, kernel);

            use crate::sample::Sample;
            let final_pixels = image::Luma::<u8>::unflatten(&scaled);

            /*
            let _filtered_image = image::ImageBuffer::from_fn(
                scaled_width, scaled_height, |x, y| {
                    let index = (y * scaled_width + x) as usize;
                    final_pixels[index]
                }
            );

            _filtered_image.save("/home/sen/Projects/RS170/ntsc_test.png").unwrap();
            */
        }
    }

    #[test]
    fn it_preprocesses_for_ntsc() {
        /*
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
        */
    }

    #[test]
    fn it_filters_new_1d() {

        // this API sucks
        let kernel = crate::signal::moving_average_kernel_new(SignalShape::OneDimensional(3));

        let data: Vec<u8> = vec![10, 5, 10, 5, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0];
        println!("original dat:\t{:02x?}", data);

        let shape = SignalShape::OneDimensional(data.len());
        let signal = Signal::<Vec<f32>>::from_samples(shape, &data);
        let filtered_data = signal.filter(&kernel);
        println!("filtered new:\t{:02x?}", filtered_data.data);

        let data: Vec<Rgb<u8>> = vec![Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255]), Rgb([0,0,0]), 
                                      Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255]), Rgb([0,0,0]), 
                                      Rgb([255,255,255]), Rgb([0,0,0]), Rgb([255,255,255])];
        println!("original dat:\t{:02x?}", data);

        let shape = SignalShape::OneDimensional(data.len());
        let signal = Signal::<Vec<f32>>::from_samples(shape, &data);
        let filtered_data = signal.filter(&kernel).data;

        println!("filtered new:\t{:02x?}", filtered_data);
    }

    #[test]
    fn it_filters_new_2d() {
        let kernel = crate::signal::moving_average_kernel_new(SignalShape::TwoDimensional(3, 3));

        let data = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(7, 7),
            vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
                 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22,
                 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
                 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30].as_slice()
        );
        println!("original:\n{}", data);
        let filtered = data.filter(&kernel);
        println!("filtered:\n{}", filtered);

        let data = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(7, 5),
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00].as_slice()
        );
        println!("original:\n{}", data);
        let filtered: Signal<Vec<f32>> = data.filter(&kernel);
        println!("filtered:\n{}", filtered);

        let data = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(6, 6),
            vec![
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            ].as_slice()
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

        let kernel = Kernel::from_samples(
            SignalShape::OneDimensional(10),
            moving_average_kernel(10).as_slice()
        );

        for i in 0..height {
            let row_start =  i as usize * width * 3;
            let row_end = row_start + (width * 3);

            let row_bytes = &mut pixels[row_start..row_end];
            let mut row_pixels: &mut [Rgb<u8>] = unsafe {
                std::slice::from_raw_parts_mut(row_bytes.as_mut_ptr() as *mut Rgb<u8>, width)
            };

            let row_pixel_len = row_pixels.len();

            Signal::<Vec<f32>>::from_samples(
                SignalShape::OneDimensional(row_pixel_len),
                &mut row_pixels,
            ).filter_in_place::<Vec<f32>>(&kernel);
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

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(img.width() as usize, img.height() as usize),
            &pixels
        );

        let sampled = signal.resample(2.0, SamplingMethod::Bilinear);
        use crate::sample::Sample;
        let final_pixels = Rgb::<u8>::unflatten(&sampled);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                final_pixels[index]
            }
        );
        sampled_image.save("/home/sen/Projects/RS170/double.png").unwrap();

        let scale = 0.5;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let sampled = signal.resample(scale, SamplingMethod::Bilinear);
        let final_pixels = Rgb::<u8>::unflatten(&sampled);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                final_pixels[index]
            }
        );
        sampled_image.save("/home/sen/Projects/RS170/half.png").unwrap();
    }

    #[test]
    fn it_resamples_3d_luma() {
        use crate::sample::Sample;
        let filename = "/home/sen/Projects/RS170/PM5644.png";
        let img_ = ImageReader::open(filename).unwrap().decode().unwrap();
        let img = img_.to_luma8();

        img.save("/home/sen/Projects/RS170/luma_orig.png").unwrap();
        let scale = 2.0;
        let scaled_width = (img.width() as f32 * scale) as u32;
        let scaled_height = (img.height() as f32 * scale) as u32;

        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<image::Luma<u8>>>();

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(img.width() as usize, img.height() as usize),
            &pixels
        );

        let sampled = signal.resample(2.0, SamplingMethod::Bilinear);
        let final_pixels = image::Luma::<u8>::unflatten(&sampled);

        let sampled_image = image::ImageBuffer::from_fn(
            scaled_width, scaled_height, |x, y| {
                let index = (y * scaled_width + x) as usize;
                final_pixels[index]
            }
        );

        sampled_image.save("/home/sen/Projects/RS170/luma_resample.png").unwrap();
    }

    #[test]
    fn it_resamples_2d() {
        let shape = SignalShape::OneDimensional(10);
        let data: [u8; 10] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
        let signal = Signal::<Vec<f32>>::from_samples(shape, &data);
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

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::OneDimensional(7),
            vec![1, 3, 5, 7, 9, 11, 13].as_slice()
        );

        let sampled = signal.resample(2.0, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::OneDimensional(7),
            vec![1, 1, 90, 2, 100, 22, 13, 1].as_slice()
        );

        let sampled = signal.resample(2.0, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::OneDimensional(7),
            vec![1, 1, 90, 2, 100, 22, 13, 1].as_slice()
        );

        let sampled = signal.resample(0.732, SamplingMethod::Linear);
        println!("original: {:?}", signal.data);
        println!("resample: {:?}", sampled.data);

        let mut f = File::open("/home/sen/Projects/RS170/sine_wave_u8.bin").unwrap();
        let mut data = Vec::<u8>::new();
        f.read_to_end(&mut data).unwrap();

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::OneDimensional(data.len()),
            &data
        );

        use crate::sample::Sample;
        let sampled = u8::unflatten(&signal.resample(0.5, SamplingMethod::Linear));
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_downscaled_0.5.bin").unwrap();
        f2.write_all(sampled.as_slice()).unwrap();

        let signal = Signal::<Vec<f32>>::from_samples(
            SignalShape::OneDimensional(data.len()),
            &mut data
        );
        let sampled = u8::unflatten(&signal.resample(5.0, SamplingMethod::Linear));
        let mut f2 = File::create("/home/sen/Projects/RS170/sine_wave_upscaled_5.bin").unwrap();
        f2.write_all(sampled.as_slice()).unwrap();
    }

    #[test]
    fn it_resamples_3d() {
        let data = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(7, 7),
            vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
                 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22,
                 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
                 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30].as_slice()
        );

        println!("original:\n{}", data);
        let sampled = data.resample(0.5, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);
        let sampled = data.resample(2.0, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);

        let data = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(7, 5),
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00].as_slice()
        );
        println!("original:\n{}", data);
        let sampled = data.resample(0.5, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);
        let sampled = data.resample(2.0, SamplingMethod::Bilinear);
        println!("sampled :\n{}", sampled);

        let data = Signal::<Vec<f32>>::from_samples(
            SignalShape::TwoDimensional(6, 6),
            vec![
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
                Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]), Rgb([255,255,255]),
            ].as_slice()
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
        assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.50), 20.0);
        assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.25), 17.5);
        assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.75), 22.5);
        assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.01), 15.1);
        assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.99), 24.9);
        // figure out how much precision is necessary. This works but fails due to
        // precision error
        // assert_eq!(linear_interpolate(p0.x, p0.y, p1.x, p1.y, 1.33333), 18.33333);
    }

    #[test]
    fn it_interpolates_bilinear() {
        assert_eq!(bilinear_interpolate(0.0, 0.0, 1.0, 1.0, 100.0, 200.0, 300.0, 100.0, 0.5, 0.5), 175.0);
        assert_eq!(bilinear_interpolate(0.0, 0.0, 1.0, 1.0, 100.0, 200.0, 300.0, 100.0, 0.2, 0.8), 232.0);
        assert_eq!(bilinear_interpolate(0.0, 0.0, 1.0, 1.0, 100.0, 200.0, 300.0, 100.0, 0.231, 0.587), 199.8209);

        println!("{}", bilinear_interpolate(1.0, 0.0, 2.0, 1.0, 255.0, 0.0, 76.0, 255.0, 1.0, 1.5));

    }

    #[test]
    fn resample_1d_1m_samples() {
        use crate::sample::Sample;
        let mut f = File::open("/home/sen/Projects/RS170/sawtooth.bin").unwrap();
        let mut data = Vec::<u8>::new();
        f.read_to_end(&mut data).unwrap();

        let mut data = data.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let signal = Signal::new(SignalShape::OneDimensional(data.len()), 1, &mut data);
        for _ in 0..10 {
            // for downsampling, scale 0.1
            // Linear can do roughly 143B samples per second
            // Point can do roughly 250B samples per second
            // Point is 43% faster - is this warrented?
            
            // for upsampling, scale 10.0
            // Point can do 2.7M samples per second
            // Linear can do roughly roughly 1.6M samples per second
            // Point ~41% faster
            
            // upsampling, scale 1.1
            // linear can do roughly 10.6M samples per second
            // point can do roughly 15.9M samples per second
            // point is roughly 50% faster
            let _resampled = signal.resample(10.0, SamplingMethod::Linear);
        }
        // let resampled = signal.resample(11.1, SamplingMethod::Linear);

        // let mut f = File::create("/home/sen/Projects/RS170/sawtooth_downscaled_0.1.bin").unwrap();
        // f.write_all(&u8::unflatten(&resampled)).unwrap();

        // let mut f = File::open("/home/sen/Projects/RS170/sawtooth.bin").unwrap();
        // let mut data = Vec::<u8>::new();
        // f.read_to_end(&mut data).unwrap();

        // for _ in 0..1000 {
            // let signal = Signal::<&mut [f32]>::from_samples(SignalShape::OneDimensional(data.len()), &data);
            // let resampled = signal.resample(0.1, SamplingMethod::Linear);
        // }

        // let mut f = File::create("/home/sen/Projects/RS170/sawtooth_downscaled_0.1.bin").unwrap();
        // f.write_all(&u8::unflatten(&resampled)).unwrap();
    }

    #[test]
    fn it_resamples_with_depth_1d() {
        let signal = rgb8_to_signal_1d("/home/sen/Projects/RS170/1d_image_simple2.png");
        let resampled = signal._resample(5.0, SamplingMethod::Linear);
        let img = signal_1d_to_rgb8(resampled);
        img.save("../1d_resampled.png").unwrap();
    }

    #[test]
    fn it_resamples_with_depth_2d() {
        let filename = "/home/sen/Projects/RS170/2d_image_simple_stripe.png";
        let signal = rgb8_to_signal_2d(filename);
        let resampled = signal._resample(5.0, SamplingMethod::Bilinear);
        let img = signal_2d_to_rgb8(resampled);
        img.save("../2d_resampled.png").unwrap();
    }

    use crate::sample::Sample;
    fn rgb8_to_signal_1d(image_filepath: &str) -> Signal<Vec<f32>> {
        let img_ = ImageReader::open(image_filepath).unwrap().decode().unwrap();
        let img = img_.to_rgb8();
        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>();
        let pixels = Rgb::<u8>::flatten(&pixels);
        Signal::new(SignalShape::OneDimensional(pixels.len()), 3, pixels)
    }

    fn signal_1d_to_rgb8(signal: Signal<Vec<f32>>) -> image::RgbImage {
        let final_pixels = Rgb::<u8>::unflatten(&signal);
        let length = final_pixels.len();
        image::ImageBuffer::from_fn(
            length as u32, 1, |x, y| {
                let index = (y * 1 + x) as usize;
                final_pixels[index]
            }
        )
    }
    fn rgb8_to_signal_2d(image_filepath: &str) -> Signal<Vec<f32>> {
        let img_ = ImageReader::open(image_filepath).unwrap().decode().unwrap();
        let img = img_.to_rgb8();
        let pixels = img.enumerate_pixels().map(|(_, _, pixel)| *pixel).collect::<Vec<Rgb<u8>>>();
        let pixels = Rgb::<u8>::flatten(&pixels);
        Signal::new(SignalShape::TwoDimensional((img.width() * 3) as usize, img.height() as usize), 3, pixels)
    }

    fn signal_2d_to_rgb8(signal: Signal<Vec<f32>>) -> image::RgbImage {
        let (width, height) = if let SignalShape::TwoDimensional(width, height) = signal.shape {
            (width / signal.sample_depth, height)
        } else {
            panic!()
        };
        let final_pixels = Rgb::<u8>::unflatten(&signal);
        image::ImageBuffer::from_fn(
            width as u32, height as u32, |x, y| {
                let index = y as usize * width + x as usize;
                final_pixels[index]
            }
        )
    }
}
