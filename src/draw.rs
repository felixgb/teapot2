pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

const WIDTH: usize = 500;
const HEIGHT: usize = 500;
pub fn draw_point(x: usize, y: usize, point: &Color, data: &mut [u8]) {
    // invert the image
    // let base = 4 * ((500 * (499 - y)) + x);
    if x < WIDTH && x > 0 && y < HEIGHT && y > 0 {
        let base = 4 * ((HEIGHT * y) + x);
        data[(base + 0)] = point.r;
        data[(base + 1)] = point.g;
        data[(base + 2)] = point.b;
        data[(base + 3)] = point.a;
    }
}
