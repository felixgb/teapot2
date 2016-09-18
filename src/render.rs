// use bresenham::Bresenham;
use draw;
use na::{Vector2, Vector3, Vector4, Matrix4, cross, norm, normalize, dot, new_identity, PerspectiveMatrix3};
use std::cmp;
use std::f32;
use std::f64;

use std::{thread, time};
use std::io::prelude::*;
use std::sync::mpsc;

use flate2::Compression;
use flate2::write::ZlibEncoder;

const WIDTH: f32 = 500.0;
const HEIGHT: f32 = 500.0;
const DEPTH: f32 = 255.0;
const SIZE: usize = (WIDTH as usize) * (HEIGHT as usize) * 4;

pub type Triangle = Vec<Vector3<f32>>;

pub fn run(tx: mpsc::Sender<Vec<u8>>, inp_rx: mpsc::Receiver<Vec<u8>>, tris: &Vec<Triangle>) {

    let mut light_dir = normalize(&Vector3::new(1.0, 0.0, -1.0));

    let mut wtc: Matrix4<f32> = new_identity(4);
    wtc[(3, 1)] = -10.0;
    wtc[(3, 2)] = -20.0;

    let fov = 45.0;
    let near = 0.1;
    let far = DEPTH;
    let proj = build_proj_matrix(fov, near, far);

    let mut screen;
    let mut zbuf;

    let mut world_coords = vec![Vector3::new(0.0, 0.0, 0.0); 3];
    let mut screen_coords = vec![Vector3::new(0.0, 0.0, 0.0); 3];

    let up = Vector3::new(0.0, 1.0, 0.0);
    let mut pos = Vector3::new(0.0, 0.0, 0.0);
    let mut front = Vector3::new(0.0, 0.0, -1.0);

    let mut last_x = WIDTH as i32 / 2;
    let mut last_y = HEIGHT as i32 / 2;

    // front = normalize(&front);

    let mut intensity = vec![0.0; 3];
    let mut size = 1.0;

    let sensitivity: f32 = 0.5;
    let cam_speed: f32 = 0.1;
    let mut yaw: f32 = -90.0;
    let mut pitch: f32 = 0.0;
    loop {
        thread::sleep(time::Duration::from_millis(16));
        screen = vec![50; SIZE];
        zbuf = vec![-f32::MAX; SIZE];

        if let Ok(codes) = inp_rx.try_recv() {
            let (mouse_pos_codes, key_codes) = codes.split_at(4);
            let mouse_x: u16 = mouse_pos_codes[0] as u16 | (mouse_pos_codes[1] as u16) << 8;
            let mouse_y: u16 = mouse_pos_codes[2] as u16 | (mouse_pos_codes[3] as u16) << 8;

            let x_offset = (mouse_x as i32 - last_x) as f32 * sensitivity;
            let y_offset = (last_y - mouse_y as i32) as f32 * sensitivity;
            last_x = mouse_x as i32;
            last_y = mouse_y as i32;
            yaw += x_offset;
            pitch += y_offset;

            if pitch > 89.0 { pitch =  89.0 }
            if pitch < -89.0 { pitch = -89.0 }

            front.x = pitch.to_radians().cos() * yaw.to_radians().cos();
            front.y = pitch.to_radians().sin();
            front.z = pitch.to_radians().cos() * yaw.to_radians().sin();
            front = normalize(&front);

            for &code in key_codes {
                match code {
                    // 87 => pos.z += 0.05, // w
                    // 83 => pos.z -= 0.05, // s
                    87 => pos = pos + (cam_speed * front), // w
                    83 => pos = pos - (cam_speed * front), // s
                    // 68 => pos.x -= 0.05, // d
                    // 65 => pos.x += 0.05, // a
                    65 => pos = pos - normalize(&cross(&front, &up)) * cam_speed,
                    68 => pos = pos + normalize(&cross(&front, &up)) * cam_speed,
                    // 81 => pos.y += 0.05, // q
                    // 69 => pos.y -= 0.05, // e
                    // 38 => dir.x += 0.05, // up
                    // 40 => dir.x -= 0.05, // down
                    // 37 => dir.y -= 0.05, // left
                    // 39 => dir.y += 0.05, // left
                    // 38 => pitch += 0.5, // up
                    // 40 => pitch -= 0.5, // down
                    // 37 => yaw -= 0.5, // left
                    // 39 => yaw += 0.5, // left
                    // 90 => size += 0.005, // right
                    // 88 => size -= 0.005, // right
                    _ => (),
                };
                // pos.y = 0.0;
            }
        }

        let center = Vector3::new(0.0, 0.0, 0.0);
        let view = lookat(pos, pos + front, up);
        let tran_mat = translate(Vector3::new(0.0, 0.0, -3.0));
        let rot_mat = dir_to_transforms(Vector3::new(0.0, 0.0, 0.0));

        for tri in tris {
            for (i, v) in tri.iter().enumerate() {
                // let transed = dir_mat * (trans * v2m(v));
                // let transed = my_mul(v2m(v), rot_mat);
                // let rotted = my_mul(transed, tran_mat);
                // let viewed = my_mul(rotted, view);
                let viewed = view * (rot_mat * (tran_mat * v2m(v)));
                let v_cam = my_mul(viewed, wtc);
                let v_proj = my_mul(v_cam, proj);
                screen_coords[i] = to_raster(m2v(v_proj));
                world_coords[i] = *v;
            }

            let a = world_coords[2] - world_coords[0];
            let b = world_coords[1] - world_coords[0];

            let mut n = cross(&a, &b);
            n = normalize(&n);

            let mut intensity = dot(&n, &light_dir);
            if intensity < 0.0 {
                intensity = 0.05;
            }
            let shade = draw::Color {
                r: (intensity * 255.0) as u8,
                g: (intensity * 255.0) as u8,
                b: (intensity * 255.0) as u8,
                a: 255
            };
            triangle(&screen_coords, zbuf.as_mut_slice(), &shade, screen.as_mut_slice());
        }

        let mut e = ZlibEncoder::new(Vec::new(), Compression::Default);

        e.write(screen.as_slice());

        let bytes = e.finish().unwrap();

        tx.send(bytes).unwrap();
    }
}

// type Plane = Vec<Vector3<f32>>;
type Frustum = Vec<Plane>;

struct Plane {
    normal: Vector3<f32>,
    d: f32,
}

impl Plane {
    fn normalize(&mut self) {
        let v_len = vec_len(self.normal);
        self.normal /= v_len;
        self.d /= v_len;
    }
}

// creates viewing frustum from view-projection matrix
fn get_frustum(mat: Matrix4<f32>) -> Vec<Plane> {
    let mut planes: Vec<Plane> = Vec::new();
    for i in 0..6 {
        let mut p = Plane { normal: Vector3::new(0.0, 0.0, 0.0), d: 0.0 };
        planes.push(p);
    }
    planes[0].normal.x = mat[(0, 3)] + mat[(0, 0)];
    planes[0].normal.y = mat[(1, 3)] + mat[(1, 0)];
    planes[0].normal.z = mat[(2, 3)] + mat[(2, 0)];
    planes[0].d        = mat[(3, 3)] + mat[(3, 0)];
    planes[1].normal.x = mat[(0, 3)] - mat[(0, 0)];
    planes[1].normal.y = mat[(1, 3)] - mat[(1, 0)];
    planes[1].normal.z = mat[(2, 3)] - mat[(2, 0)];
    planes[1].d        = mat[(3, 3)] - mat[(3, 0)];
    planes[2].normal.x = mat[(0, 3)] + mat[(0, 1)];
    planes[2].normal.y = mat[(1, 3)] + mat[(1, 1)];
    planes[2].normal.z = mat[(2, 3)] + mat[(2, 1)];
    planes[2].d        = mat[(3, 3)] + mat[(3, 1)];
    planes[3].normal.x = mat[(0, 3)] - mat[(0, 1)];
    planes[3].normal.y = mat[(1, 3)] - mat[(1, 1)];
    planes[3].normal.z = mat[(2, 3)] - mat[(2, 1)];
    planes[3].d        = mat[(3, 3)] - mat[(3, 1)];
    planes[4].normal.x = mat[(0, 3)] + mat[(0, 2)];
    planes[4].normal.y = mat[(1, 3)] + mat[(1, 2)];
    planes[4].normal.z = mat[(2, 3)] + mat[(2, 2)];
    planes[4].d        = mat[(3, 3)] + mat[(3, 2)];
    planes[5].normal.x = mat[(0, 3)] - mat[(0, 2)];
    planes[5].normal.y = mat[(1, 3)] - mat[(1, 2)];
    planes[5].normal.z = mat[(2, 3)] - mat[(2, 2)];
    planes[5].d        = mat[(3, 3)] - mat[(3, 2)];

    let planes2 = planes.clone();
    for plane in planes2 {
        plane.normalize();
    }

    planes
}

fn vec_len(vec: Vector3<f32>) -> f32 {
    (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z).sqrt()
}
//
//fn get_frustum(fov: f32, ratio: f32, near: f32, far: f32, pos: Vector3<f32>, dir: Vector3<f32>) -> Frustum {
//    let up = normalize(&Vector3::new(0.0, 1.0, 0.0));
//    let norm_dir = normalize(&dir);
//    let tang = 2.0 * (fov / 2.0).to_radians().tan();
//    let nh = near * tang;
//    let nw = nh * ratio;
//    let fh = far * tang;
//    println!("FAR HEIGHT: {}", fh);
//    let fw = fh * ratio;
//
//    // let z = normalize(&(pos - norm_dir));
//    println!("norm... {}", Vector3::new(0.0, 0.0, 0.0));
//    let z = Vector3::new(0.0, 0.0, -1.0);
//
//    let x = Vector3::new(1.0, 0.0, 0.0);
//
//    let y = Vector3::new(0.0, 1.0, 0.0);
//
//
//    let nc = pos - dir * near;
//    let fc = pos - dir * far;
//
//    println!("up: {}", up);
//    println!("z: {}", z);
//    println!("up * z: {}", (up * z));
//
//    let ntl = nc + y * nh - x * nw;
//    let ntr = nc + y * nh + x * nw;
//    let nbl = nc - y * nh - x * nw;
//    let nbr = nc - y * nh + x * nw;
//
//    let ftl = fc + y * fh - x * fw;
//    let ftr = fc + y * fh + x * fw;
//    let fbl = fc - y * fh - x * fw;
//    let fbr = fc - y * fh + x * fw;
//
//    vec![
//        vec![ntr, ntl, ftl], // top
//        vec![nbl, nbr, fbr], // bottom
//        vec![ntl, nbl, fbl], // left
//        vec![nbr, ntr, fbr], // right
//        vec![ntl, ntr, nbr], // near
//        vec![ftr, ftl, fbl], // far
//    ]
//}
//
fn plane_dist_signed(plane: Plane, pt: Vector3<f32>) -> f32 {
    dot(&plane.normal, &pt) + plane.d
    // let v = plane[1] - plane[0];
    // let u = plane[2] - plane[0];
    // let n = normalize(&cross(&v, &u));
    // let d = dot(&-n, &plane[0]);
    // dot(&n, &point) + d
}
//
//fn is_in(tri: &Triangle, frustum: &Frustum) -> bool {
//    for plane in frustum {
//        let mut out_count = 0;
//        let mut in_count = 0;
//        for point in tri {
//            // println!("plane dist for point {}: {}", *point, plane_dist(&plane, *point));
//            if plane_dist(&plane, *point) < 0.0 {
//                out_count += 1;
//            } else {
//                in_count += 1;
//            }
//            if !(in_count != 0) {
//                return false;
//            } else if out_count != 0 {
//                return true;
//            }
//        }
//    }
//    return true;
//}

fn point_in_frustum(pt: Vector3<f32>, frustum: &Vec<Plane>) -> bool {
    for plane in frustum {
        if plane_dist_signed(*plane, pt) < 0.0 {
            return false;
        }
    }
    true
}

fn tri_in_frustum(tri: Triangle, frustum: &Vec<Plane>) -> bool {
    tri.iter().any(| &pt | point_in_frustum(pt, frustum))
}

#[cfg(test)]
mod test {

    use na::{Vector3};
    use render;
//
//    #[test]
//    fn test_plane_dist() {
//        let plane = vec![
//            Vector3::new(1.0, 0.0, 1.0),
//            Vector3::new(2.0, 0.0, 0.0),
//            Vector3::new(1.0, 0.0, 0.0),
//        ];
//        let point = Vector3::new(0.0, 1.0, 0.0);
//        assert_eq!(render::plane_dist_signed(&plane, point), 1.0);
//        let point = Vector3::new(0.0, -1.0, 0.0);
//        assert_eq!(render::plane_dist_signed(&plane, point), -1.0);
//        let point = Vector3::new(0.0, 0.0, 0.0);
//        assert_eq!(render::plane_dist(&plane, point), 0.0);
//    }
//
    #[test]
    fn test_get_frustum() {
        let pos = Vector3::new(0.0, 0.0, 0.0);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let foo: Vec<render::Plane> = Vec::new();
        // assert!(render::get_frustum(45.0, 1.0, 0.1, 255.0, pos, dir) == foo);
    }

    #[test]
    fn test_is_in() {
        let up = Vector3::new(0.0, 1.0, 0.0);
        let pos = Vector3::new(0.0, 0.0, 0.0);
        let dir = Vector3::new(0.0, 0.0, -1.0);
        let fov = 45.0;
        let near = 0.1;
        let far = 255.0;
        let proj = render::build_proj_matrix(fov, near, far);
        let view = render::lookat(pos, pos + dir, up);
        let in_tri = vec![
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, -1.0),
        ];
        let frustum = render::get_frustum(view * proj);
        assert!(render::tri_in_frustum(in_tri, &frustum));
    }

    #[test]
    fn test_not_is_in() {
//        let pos = Vector3::new(0.0, 0.0, 0.0);
//        let dir = Vector3::new(0.0, 0.0, -1.0);
//        let out_tri = vec![
//            Vector3::new(0.0, 0.0, 1.0),
//            Vector3::new(1.0, 0.0, 1.0),
//            Vector3::new(0.0, 1.0, 1.0),
//        ];
//        let frustum = render::get_frustum(45.0, 1.0, 0.1, 255.0, pos, dir);
//        // println!("frustum: {:#?}", frustum);
//        assert!(!render::is_in(&out_tri, &frustum));
    }
}

fn my_mul(v: Vector4<f32>, m: Matrix4<f32>) -> Vector4<f32> {
    let mut tmp = m * v;
    if tmp.w != 1.0 {
        tmp.x /= tmp.w;
        tmp.y /= tmp.w;
        tmp.z /= tmp.w;
    }
    tmp
}

/*
fn to_raster(p: Vector3<f32>) -> Vector3<f32> {
    let x = cmp::min((WIDTH - 1.0) as i32, ((p.x + 1.0) * 0.5 * WIDTH) as i32);
    let y = cmp::min((HEIGHT - 1.0) as i32, ((1.0 - (p.y + 1.0) * 0.5) * HEIGHT) as i32);
    Vector3::new(x as f32, y as f32, p.z)
}
*/

fn to_raster(p: Vector3<f32>) -> Vector3<f32> {
    let x = ((p.x + 1.0) * 0.5 * WIDTH) as i32;
    let y = ((1.0 - (p.y + 1.0) * 0.5) * HEIGHT) as i32;
    Vector3::new(x as f32, y as f32, p.z)
}

fn build_proj_matrix(fov: f32, near: f32, far: f32) -> Matrix4<f32> {
    let scale = 1.0 / (fov * 0.5 * f32::consts::PI / 180.0).tan();
    let mut m: Matrix4<f32> = new_identity(4);
    m[(0, 0)] = scale;
    m[(1, 1)] = scale;
    m[(2, 2)] = -far / (far - near);;
    m[(2, 3)] = -far * near / (far - near);
    m[(3, 2)] = -1.0;
    m[(3, 3)] = 0.0;
    m
}

fn min3(a: i32, b: i32, c: i32) -> i32 {
    cmp::min(a, cmp::min(b, c))
}

fn max3(a: i32, b: i32, c: i32) -> i32 {
    cmp::max(a, cmp::max(b, c))
}

fn triangle(pts: &Vec<Vector3<f32>>, zbuf: &mut [f32], color: &draw::Color, image: &mut [u8]) {
    let mut min_x = min3(pts[0].x as i32, pts[1].x as i32, pts[2].x as i32);
    let mut min_y = min3(pts[0].y as i32, pts[1].y as i32, pts[2].y as i32);
    let mut max_x = max3(pts[0].x as i32, pts[1].x as i32, pts[2].x as i32);
    let mut max_y = max3(pts[0].y as i32, pts[1].y as i32, pts[2].y as i32);

    min_x = cmp::max(min_x, 0);
    min_y = cmp::max(min_y, 0);
    max_x = cmp::min(max_x, WIDTH as i32 - 1);
    max_y = cmp::min(max_y, HEIGHT as i32 - 1);

    let mut point = Vector3::new(0.0, 0.0, 0.0);

    for x in min_x..max_x + 1 {
        point.x = x as f32;
        for y in min_y..max_y + 1 {
            point.y = y as f32;
            let bc_screen = barycentric(pts, point);
            if bc_screen.x < 0.0 || bc_screen.y < 0.0 || bc_screen.z < 0.0 {
                continue;
            } else {
                point.z = 0.0;
                point.z += pts[0].z * bc_screen.x;
                point.z += pts[1].z * bc_screen.y;
                point.z += pts[2].z * bc_screen.z;
                if zbuf[(point.x + point.y * WIDTH) as usize] < point.z {
                    zbuf[(point.x + point.y * WIDTH) as usize] = point.z;
                    draw::draw_point(point.x as usize, point.y as usize, color, image);
                }
            }
        }
    }
}

/*
fn triangle(pts: &Vec<Vector3<f32>>, zbuf: &mut [f32], color: &draw::Color, image: &mut [u8]) {
    let mut bbox_min = Vector2::new(WIDTH as i32, HEIGHT as i32);
    let mut bbox_max = Vector2::new(0, 0);
    let clamp = Vector2::new(WIDTH as i32, HEIGHT as i32);

    // really bad bounding box code
    for point in pts {
        bbox_min.x = cmp::max(0, cmp::min(bbox_min.x, point.x as i32));
        bbox_max.x = cmp::min(clamp.x, cmp::max(bbox_max.x, point.x as i32));
        bbox_min.y = cmp::max(0, cmp::min(bbox_min.y, point.y as i32));
        bbox_max.y = cmp::min(clamp.y, cmp::max(bbox_max.y, point.y as i32));
    }

    let mut point = Vector3::new(0.0, 0.0, 0.0);

    for x in bbox_min.x..(bbox_max.x) {
        point.x = x as f32;
        for y in bbox_min.y..(bbox_max.y) {
            point.y = y as f32;
            let bc_screen = barycentric(pts, point);
            if bc_screen.x < 0.0 || bc_screen.y < 0.0 || bc_screen.z < 0.0 {
                continue;
            } else {
                point.z = 0.0;
                point.z += pts[0].z * bc_screen.x;
                point.z += pts[1].z * bc_screen.y;
                point.z += pts[2].z * bc_screen.z;
                if zbuf[(point.x + point.y * WIDTH) as usize] < point.z {
                    zbuf[(point.x + point.y * WIDTH) as usize] = point.z;
                    draw::draw_point(point.x as usize, point.y as usize, color, image);
                }
            }
        }
    }
}
*/

fn dir_to_transforms(dir: Vector3<f32>) -> Matrix4<f32> {
    rot_y(dir.y) * rot_x(dir.x)
}

/*
fn move_dir(old: Vector3<f32>, code: u8) -> Vector3<f32> {
    match code {
        37 => Vector3::new(old.x - 0.01, old.y, old.z),
        38 => Vector3::new(old.x, old.y, old.z + 0.01),
        39 => Vector3::new(old.x + 0.01, old.y, old.z),
        40 => Vector3::new(old.x, old.y, old.z - 0.01),
        _ => old,
    }
}
*/

// https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling
// http://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/rasterization-stage
fn barycentric(pts: &Vec<Vector3<f32>>, point: Vector3<f32>) -> Vector3<f32> {
    let a1 = Vector3::new(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - point.x);
    let a2 = Vector3::new(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - point.y);
    let u = cross(&a1, &a2);
    if u.z.abs() < 1.0 {
        Vector3::new(-1.0, 1.0, 1.0)
    } else {
        Vector3::new(1.0 - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z)
    }
}

fn world_to_screen(v: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(
        ((v.x + 1.0) * WIDTH / 2.0 + 0.5).round(),
        ((v.y + 1.0) * HEIGHT / 2.0 + 0.5).round(),
        v.z,
        )
}

fn m2v(v: Vector4<f32>) -> Vector3<f32> {
    Vector3::new(
        v.x,
        v.y,
        v.z,
        )
}

fn v2m(v: &Vector3<f32>) -> Vector4<f32> {
    Vector4::new(
        v.x,
        v.y,
        v.z,
        1.0,
        )
}

fn viewport(x: i32, y: i32, w: i32, h: i32) -> Matrix4<f32> {
    let mut m: Matrix4<f32> = new_identity(4);
    m.m14 = (x + w) as f32 / 2.0;
    m.m24 = (y + h) as f32 / 2.0;
    m.m34 = DEPTH / 2.0;

    m.m11 = WIDTH / 2.0;
    m.m22 = HEIGHT / 2.0;
    m.m33 = DEPTH / 2.0;
    m
}

fn lookat(eye: Vector3<f32>, center: Vector3<f32>, up: Vector3<f32>) -> Matrix4<f32> {
    let z = normalize(&(eye - center));
    let x = normalize(&cross(&up, &z));
    let y = cross(&z, &x);

    let res = Matrix4::new(
        x.x, x.y, x.z, 0.0,
        y.x, y.y, y.z, 0.0,
        z.x, z.y, z.z, 0.0,
        0.0, 0.0, 0.0, 1.0
        );
    res * translate(-eye)
}

fn translate(dir: Vector3<f32>) -> Matrix4<f32> {
    let mut m: Matrix4<f32> = new_identity(4);
    m.m14 = dir.x;
    m.m24 = dir.y;
    m.m34 = dir.z;
    m
}

fn scale(sc: f32) -> Matrix4<f32> {
    let mut m: Matrix4<f32> = new_identity(4);
    m.m11 = sc;
    m.m22 = sc;
    m.m33 = sc;
    m
}

fn rot_x(theta: f32) -> Matrix4<f32> {
    let mut m: Matrix4<f32> = new_identity(4);
    m.m22 = theta.cos();
    m.m23 = -theta.sin();
    m.m33 = theta.cos();
    m.m32 = theta.sin();
    m
}

fn rot_y(theta: f32) -> Matrix4<f32> {
    let mut m: Matrix4<f32> = new_identity(4);
    m.m11 = theta.cos();
    m.m13 = theta.sin();
    m.m31 = -theta.sin();
    m.m33 = theta.cos();
    m
}

/*
pub fn line4(x0: i32, y0: i32, x1: i32, y1: i32, color: &draw::Color, image: &mut [u8]) {
    for (x, y) in Bresenham::new((x0 as isize, y0 as isize), (x1 as isize, y1 as isize)) {
        draw::draw_point(x as usize, y as usize, color, image);
    }
}
*/

/*
fn triangle(tri: &Vec<Vector3<i32>>, color: &draw::Color, image: &mut [u8]) {
    let mut bbox_min = Vector2::new(500 - 1, 500 - 1);
    let mut bbox_max = Vector2::new(0, 0);
    let clamp = Vector2::new(500 - 1, 500 - 1);

    // really bad bounding box code
    for point in tri {
        bbox_min.x = cmp::max(0, cmp::min(bbox_min.x, point.x));
        bbox_max.x = cmp::min(clamp.x, cmp::max(bbox_max.x, point.x));
        bbox_min.y = cmp::max(0, cmp::min(bbox_min.y, point.y));
        bbox_max.y = cmp::min(clamp.y, cmp::max(bbox_max.y, point.y));
    }

    let mut point = Vector3::new(0.0, 0.0, 0.0);

    for x in bbox_min.x..bbox_max.x {
        point.x = x as f32;
        for y in bbox_min.y..bbox_max.y {
            point.y = y as f32;
            let bc_screen = barycentric(f_tri, point);
            if bc_screen.x < 0.0 || bc_screen.y < 0.0 || bc_screen.z < 0.0 {
                continue;
            } else {
                //point.z
                draw::draw_point(point.x as usize, point.y as usize, color, image);
            }
        }
    }
}
*/

/*
pub fn wireframe(obj: Object, color: &draw::Color, image: &mut [u8]) {
    println!("num verts: {}", obj.vertices.len());
    println!("num faces: {}", obj.faces.len());

    for face in obj.faces {
        let mut screen_coords = vec![Vector2::new(0, 0); 3];
        for i in 0..face.vertex_indices.len() {
            let i0 = face.vertex_indices[i].vertex_index - 1;
            let i1 = face.vertex_indices[(i + 1) % 3].vertex_index - 1;
            let v0 = &obj.vertices[i0];
            let v1 = &obj.vertices[i1];

            let x0 = (&v0.x + 1.0) * (500.0 / 2.0);
            let y0 = (&v0.y + 1.0) * (500.0 / 2.0);
            screen_coords[i] = Vector2::new(x0 as i32, y0 as i32);
            
            let x1 = (&v1.x + 1.0) * (500.0 / 2.0);
            let y1 = (&v1.y + 1.0) * (500.0 / 2.0);

           // line4(x0 as i32, y0 as i32, x1 as i32, y1 as i32, color, image);
        }
        triangle(&screen_coords, color, image);
    }
}
*/

/*
pub fn line2(mut x0: i32, mut y0: i32, mut x1: i32, mut y1: i32, color: &draw::Color, image: &mut [u8]) {
    if x0 > x1 {
        x0 = x1;
        y0 = y1;
    }
    let dx = x1 - x0;
    let dy = y1 - y0;
    let mut err = -1.0;
    let derr = ((dy as f32) / (dx as f32)).abs();
    let mut y = y0;
    for x in x0..(x1 - 1) {
        draw::draw_point(x as usize, y as usize, color, image);
        err = err + derr;
        if err >= 0.0 {
            y = y + 1;
            err = err - 1.0;
        }
    }
}

pub fn line3(x0: i32, y0: i32, x1: i32, y1: i32, color: &draw::Color, image: &mut [u8]) {
    let mut steep = false;
    let mut m_x0 = x0;
    let mut m_y0 = y0;
    let mut m_x1 = x1;
    let mut m_y1 = y1;

    if (x0 - x1).abs() < (y0 - y1).abs() {
        m_x0 = m_y0;
        m_x1 = m_y1;
        steep = true;
    }

    if m_x0 > m_x1 {
        m_x0 = m_x1;
        m_y0 = m_y1;
    }

    for x in m_x0..(m_x1 + 1) {
        let t = ((x - m_x0) as f32) / ((m_x1 - m_x0) as f32);
        let y = (m_y0 as f32) * (1.0 - t) + (m_y1 as f32 * t);
        if steep {
            draw::draw_point(y as usize, x as usize, color, image);
        } else {
            draw::draw_point(x as usize, y as usize, color, image);
        }
    }
}

pub fn line(x0: i32, y0: i32, x1: i32, y1: i32, color: &draw::Color, image: &mut [u8]) {
    let mut m_x0 = x0;
    let mut m_y0 = y0;
    let mut m_x1 = x1;
    let mut m_y1 = y1;
    let mut steep = false;

    if (x0 - x1).abs() < (y0 - y1).abs() {
        m_x0 = y0;
        m_x1 = y1;
        steep = true;
    }

    if m_x0 > m_x1 {
        m_x0 = x1;
        m_y0 = y1;
    }

    let dx = m_x1 - m_x0;
    let dy = m_y1 - m_y0;
    let derr2 = dy.abs() * 2;
    let mut err2 = 0;
    let mut y = m_y0;

    for x in m_x0..(m_x1 + 1) {
        if steep {
            draw::draw_point(y as usize, x as usize, color, image);
        } else {
            draw::draw_point(x as usize, y as usize, color, image);
        }
        err2 += derr2;
        if err2 > dx {
            y += if m_y1 > m_y0 { 1 } else { -1 };
            err2 -= dx * 2;
        }
    }
}
*/
