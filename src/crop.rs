#[derive(Debug)]
pub enum CropDirection {
    Vertical,
    Horizontal
}

pub struct CropParams {
    pub magnitude: usize,
    pub direction: CropDirection
}

pub fn crop_symmetric_1d<S>(data: &[S], width: usize) -> &[S] {
    let offset = (data.len() - width) / 2;
    &data[offset..data.len() - offset]
}

pub fn crop_symmetric_width_2d<S: Clone>(data: &[S], width: usize, height: usize, crop_to: usize) -> (usize, usize, Vec<S>) {
    let mut cropped = Vec::<S>::new();
    for i in 0..height {
        let row_start =  i as usize * width as usize;
        let row_end = row_start + width as usize;
        let row = &data[row_start..row_end];
        cropped.append(&mut crop_symmetric_1d(row, crop_to).to_vec());
    }
    (crop_to, height, cropped)
}

pub fn crop_symmetric_2d<S: Clone>(data: &[S], width: usize, height: usize, crop_params: CropParams) -> (usize, usize, Vec<S>) {
    let mut cropped = Vec::<S>::new();

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
