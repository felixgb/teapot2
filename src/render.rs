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
                    87 => pos = pos + (cam_speed * front), // w
                    83 => pos = pos - (cam_speed * front), // s
                    65 => pos = pos - normalize(&cross(&front, &up)) * cam_speed,
                    68 => pos = pos + normalize(&cross(&front, &up)) * cam_speed,
                    _ => (),
                };
                // pos.y = 0.0;
            }
        }
        

        let center = Vector3::new(0.0, 0.0, 0.0);
        let view = lookat(pos, pos + front, up);
        let tran_mat = translate(Vector3::new(0.0, 0.0, -3.0));
        let rot_mat = dir_to_transforms(Vector3::new(0.0, 0.0, 0.0));

        let frustum = get_frustum(view * proj);

        for tri in tris {
            let transformed = model_to_world(tri, tran_mat);
            if !tri_in_frustum(&transformed, &frustum) {
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
                    a: 255,
                };
                triangle(&screen_coords, zbuf.as_mut_slice(), &shade, screen.as_mut_slice());
            } else {
                println!("not in");
            }
        }

        let mut e = ZlibEncoder::new(Vec::new(), Compression::Default);

        e.write(screen.as_slice());

        let bytes = e.finish().unwrap();

        tx.send(bytes).unwrap();
    }
}

fn model_to_world(tri: &Triangle, transforms: Matrix4<f32>) -> Triangle {
    let mut new_tri = Vec::new();
    for v in tri {
        let trans_v = m2v(transforms * v2m(v));
        new_tri.push(trans_v);
    }

    new_tri
}

struct Plane {
    normal: Vector3<f32>,
    d: f32,
}

type Frustum = Vec<Plane>;

impl Plane {
    fn normalize(&mut self) {
        let v_len = vec_len(self.normal);
        self.normal /= v_len;
        self.d /= v_len;
    }
}

// creates viewing frustum from view-projection matrix
fn get_frustum(mat: Matrix4<f32>) -> Frustum {
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

    planes[0].normalize();
    planes[1].normalize();
    planes[2].normalize();
    planes[3].normalize();
    planes[4].normalize();
    planes[5].normalize();

    planes
}

fn vec_len(vec: Vector3<f32>) -> f32 {
    (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z).sqrt()
}

// fn get_frustum(fov: f32, ratio: f32, near: f32, far: f32, pos: Vector3<f32>, dir: Vector3<f32>) -> Frustum {
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
// }

fn plane_dist_signed(plane: &Plane, pt: Vector3<f32>) -> f32 {
    dot(&plane.normal, &pt) + &plane.d
    // let v = plane[1] - plane[0];
    // let u = plane[2] - plane[0];
    // let n = normalize(&cross(&v, &u));
    // let d = dot(&-n, &plane[0]);
    // dot(&n, &point) + d
}

// fn is_in(tri: &Triangle, frustum: &Frustum) -> bool {
//    for plane in frustum {
//        let mut out_count = 0;
//        let mut in_count = 0;
//        for point in tri {
//            // println!("plane dist for point {}: {}", *point, plane_dist(&plane, *point));
//            if plane_dist_signed(plane, *point) < 0.0 {
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
// }
// 
fn point_in_frustum(pt: Vector3<f32>, frustum: &Vec<Plane>) -> bool {
    for plane in frustum {
        if plane_dist_signed(plane, pt) < 0.0 {
            return false;
        }
    }
    true
}

fn tri_in_frustum(tri: &Triangle, frustum: &Vec<Plane>) -> bool {
    (*tri).iter().any(| &pt | point_in_frustum(pt, frustum))
}

#[cfg(test)]
mod test {

    use na::{Vector3};
    use render;

    // #[test]
    // fn test_plane_dist() {
    //    let plane = vec![
    //        Vector3::new(1.0, 0.0, 1.0),
    //        Vector3::new(2.0, 0.0, 0.0),
    //        Vector3::new(1.0, 0.0, 0.0),
    //    ];
    //    let point = Vector3::new(0.0, 1.0, 0.0);
    //    assert_eq!(render::plane_dist_signed(&plane, point), 1.0);
    //    let point = Vector3::new(0.0, -1.0, 0.0);
    //    assert_eq!(render::plane_dist_signed(&plane, point), -1.0);
    //    let point = Vector3::new(0.0, 0.0, 0.0);
    //    assert_eq!(render::plane_dist(&plane, point), 0.0);
    // }

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
       let frustum = render::get_frustum(proj * view);
       assert!(render::tri_in_frustum(&in_tri, &frustum));
   }

   #[test]
   #[should_panic]
   fn test_not_is_in() {
       let up = Vector3::new(0.0, 1.0, 0.0);
       let pos = Vector3::new(0.0, 0.0, 0.0);
       let dir = Vector3::new(0.0, 0.0, -1.0);
       let fov = 45.0;
       let near = 0.1;
       let far = 255.0;
       let proj = render::build_proj_matrix(fov, near, far);
       let view = render::lookat(pos, pos + dir, up);
       let out_tri = vec![
           Vector3::new(0.0, 0.0, 1.0),
           Vector3::new(1.0, 0.0, 1.0),
           Vector3::new(0.0, 1.0, 1.0),
       ];
       let frustum = render::get_frustum(proj * view);
       assert!(render::tri_in_frustum(&out_tri, &frustum));
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

fn dir_to_transforms(dir: Vector3<f32>) -> Matrix4<f32> {
    rot_y(dir.y) * rot_x(dir.x)
}

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
