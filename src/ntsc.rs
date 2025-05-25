use crate::crop::*;
use crate::sample::*;
use crate::signal::*;
// module for generating NTSC baseband signal from input.
// starting with a single frame

const NTSC_VISIBLE_LINE_COUNT: usize = 480;

// const NTSC_EFFECTIVE_WIDTH: usize = 640;
/*
const NTSC_H_TOTAL: f32 = 63.5;
    const NTSC_H_BLANK: f32 = 10.9;
        const NTSC_H_FRONT_PORCH: f32 = 1.5;    // blanking, 0 IRE
        const NTSC_H_SYNC_TIP: f32 = 4.7;       // -40 IRE
        const NTSC_H_BREEZEWAY: f32 = 0.6;      // 0 IRE
        const NTSC_H_COLOR_BURST: f32 = 2.5;    // -20 to +20 IRE, 40 total around blanking level
        const NTSC_H_BACK_PORCH: f32 = 1.6;     // 0 IRE
const NTSC_H_ACTIVE_VIDEO: f32 = 52.6;
*/
// 10046600 Hz == 191000 ACTIVE_VIDEO sps
// const SAMPLE_RATE_HZ: usize = 10_046_600;
// const SAMPLE_PER_LINE_ACTIVE_VIDEO: usize = 191000;

fn _frame_to_signal<S: Sample>(frame: Signal<S>) {
    // for each other line in frame (split even/odd fields)
    // sample line data at appropriate sample rate (assuming a certain transmit rate)
    // convert each sample to IRE value (or voltage? implicit conversion?)
    // prepend/append front/back porches and other shit
    // concatenate all lines in the field
    // repeat for other field
    // append post-frame blanking lines
    // return full frame baseband signal
    
    let (width, _height) = if let SignalShape::TwoDimensional(width, height) = frame.shape {
        (width, height)
    } else {
        panic!("bad frame shape")
    };

    // iterate lines
    // don't forget to fix this for fields, should be every other line, then back up top, then the
    // rest
    for line in frame.data.chunks(width) {
        let _line = Signal { 
            data: line.to_vec(), 
            shape: SignalShape::OneDimensional(width) 
        };

        // scale the values in the line from S to f32 (or whatever will work best with libbladerf)
        // within the range -285.7mv to +714.3mv
        // unimplemented yet !!!
        
        // This should guarantee a specific number of samples (invariably 191000) but maybe just
        // have another function that takes a number of samples directly rather than a scale.
        // let _horizontal_vdata = line.resample(SAMPLE_PER_LINE_ACTIVE_VIDEO as f32 / NTSC_EFFECTIVE_WIDTH as f32, SamplingMethod::Linear);

        // append front porch, sync, breezeway/colorburst, back porch to line
        // push line data to frame signal
    }
}

/*
fn modulate_baseband() {
    // modulate the baseband signal as necessary
    // return modulated signal (I/Q data?)
    // possibly do this in place and send it to the bladeRF API
}
*/

pub fn crop_for_ntsc<S: Clone>(data: &[S], width: usize, height: usize) -> (usize, usize, Vec<S>) {
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

pub fn ntsc_process_frame<S: image::Pixel + Sample>(pixels: &[S], width: usize, height: usize) -> (usize, usize, Vec<S>) {
    let (cropped_width, cropped_height, pixels) = 
        crop_for_ntsc(&pixels, width as usize, height as usize);


    let scale = NTSC_VISIBLE_LINE_COUNT as f32 / cropped_height as f32;
    let scaled_width = (cropped_width as f32 * scale) as usize;
    let scaled_height = (cropped_height as f32 * scale) as usize;

    let mut scaled = Signal {
        data: pixels,
        shape: SignalShape::TwoDimensional(cropped_width, cropped_height)
    }.resample(scale, SamplingMethod::Bilinear);

    let kernel = Kernel {
        data: moving_average_kernel(3),
        shape: SignalShape::OneDimensional(7)
    };

    for y in 0..scaled_height {
        let start =  y as usize * scaled_width;
        let end = start + scaled_width;
        let row = &mut scaled.data[start..end];
        let lpf_row = Signal { data: row.to_vec(), shape: SignalShape::OneDimensional(scaled_width) };
        let lpf_row = lpf_row.filter(&kernel).data;
        for (dst, src) in row.iter_mut().zip(lpf_row.iter()) {
            *dst = *src;
        }
    }

    (scaled_width, scaled_height, scaled.data)
}
