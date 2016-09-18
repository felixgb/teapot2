use std::io::prelude::*;
use std::fs::File;
use std::f32::consts::PI;

use na::{Vector3};

use rand::{Rng, thread_rng};

use render;

const SIZE: usize = 32;
const RADIUS: usize = 10;
const SIZE_MASK: usize = SIZE as usize - 1;

fn square_to_tris(square: Vec<Vector3<f32>>) -> Vec<render::Triangle> {
    vec![
        vec![ square[0], square[2], square[3] ],
        vec![ square[1], square[2], square[0] ],
    ]
}

fn normalize(x: usize) -> f32 {
    // (1 - (-1)) / ((50 - 0) * (x - 50) + 1)
    (x as f32) / (SIZE / 10) as f32
}

fn mk_square_grid() -> Vec<Vector3<f32>> {
    let zs = noise_static();
    let mut square_grid = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let x_f = normalize(x);
            let y_f = normalize(y);
            println!("x: {}, y: {}", x_f, y_f);
            let z = zs[y * SIZE + x] * 0.7;
            square_grid.push(Vector3::new(x_f, y_f, z));
        }
    }
    square_grid
}

fn scale(v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v.x / 100.0, v.y / 100.0, v.z / 100.0)
}

fn mk_cylinder() -> Vec<Vector3<f32>> {
    let zs = noise_marble();
    let mut cylinder = Vec::new();
    for z in 0..SIZE {
        for i in 0..SIZE {
            let theta = 2.0 * PI * (i as f32) / (SIZE as f32);
            let theta_val = zs[z * SIZE + i] + theta;
            let x = ((SIZE as f32 / 2.0) + (RADIUS as f32) * theta.cos()).round();
            let y = ((SIZE as f32 / 2.0) + (RADIUS as f32) * theta.sin()).round();
            let x_val = 5.0 * zs[z * SIZE + i] + x;
            let y_val = 5.0 * zs[z * SIZE + i] + y;
            let z_val = zs[z * SIZE + i] + (z * 10) as f32;
            // let z_val = z as f32;
            cylinder.push(Vector3::new(x_val, y_val, z_val));
        }
    }

    /*
    // This is such a hack. 
    let mut close_gap = Vec::new();
    for z in 0..(SIZE - 2) {
        let row_first = z * SIZE;
        let p0 = row_first;
        let p1 = row_first + SIZE;
        let p2 = row_first + 1;
        let p3 = row_first + 1 + SIZE;
        close_gap.push(cylinder[p0]);
        close_gap.push(cylinder[p2]);
        close_gap.push(cylinder[p1]);
        close_gap.push(cylinder[p3]);
    }
    for p in close_gap {
        cylinder.push(p);
    }
    */
    cylinder
}

pub fn triangleify(square_grid: &Vec<Vector3<f32>>) -> Vec<render::Triangle> {
    let mut tris = Vec::new();
    for y in 0..(SIZE - 1) {
        for x in 0..(SIZE - 1) {
            let row_offset_even = y * SIZE;
            let row_offset_odd = (y + 1) * SIZE;
            let p0 = row_offset_even + x;
            let p1 = row_offset_even + (x + 1);
            let p2 = row_offset_odd + (x + 1);
            let p3 = row_offset_odd + x;
            tris.push(vec![
                      square_grid[p0],
                      square_grid[p2],
                      square_grid[p3],
            ]);
            tris.push(vec![
                      square_grid[p1],
                      square_grid[p2],
                      square_grid[p0],
            ]);
        }
    }
    tris
}

fn lerp(low: f32, hi: f32, t: f32) -> f32 {
    low * (1.0 - t) + hi * t
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 - t)
}

fn noise_marble() -> Vec<f32> {
    let freq = 0.5;
    let freq_mult = 1.0;
    let amp_mult = 1.0;
    let num_layers = 5;

    let mut r = vec![0.0; SIZE];
    let mut perm_table = vec![0; SIZE * 2];
    let mut noise_map = vec![0.0; SIZE * SIZE];
    let mut rng = thread_rng();

    for y in 0..SIZE {
        for x in 0..SIZE {
            for k in 0..SIZE {
                r[k] = rng.gen::<f32>();
                perm_table[k] = k;
            }

            // probably a better shuffle somewhere
            for k in 0..SIZE {
                let i = rng.gen::<usize>() & SIZE_MASK;
                let tmp = perm_table[k];
                perm_table[k] = perm_table[i];
                perm_table[i] = tmp;
                perm_table[k + SIZE] = tmp;
            }

            let mut p_noise_x = x as f32 * freq;
            let mut p_noise_y = y as f32 * freq;

            let mut amplitude = 1.0;
            let mut noise_val = 0.0;

            for i in 0..num_layers {
                noise_val += eval(p_noise_x, p_noise_y, &perm_table, &r) * amplitude;
                p_noise_x *= freq_mult;
                p_noise_y *= freq_mult;
                amplitude *= amp_mult;
            }

            noise_map[y * SIZE + x] = (((x as f32 + noise_val * 100.0) * 2.0 * PI / 200.0).sin() + 1.0) / 2.0;
        }
    }

    noise_map

}


fn noise_static() -> Vec<f32> {
    let freq = 0.02;
    let mut r = vec![0.0; SIZE];
    let mut perm_table = vec![0; SIZE * 2];
    let mut noise_map = vec![0.0; SIZE * SIZE];
    let mut rng = thread_rng();

    for y in 0..SIZE {
        for x in 0..SIZE {
            for k in 0..SIZE {
                r[k] = rng.gen::<f32>();
                perm_table[k] = k;
            }

            // probably a better shuffle somewhere
            for k in 0..SIZE {
                let i = rng.gen::<usize>() & SIZE_MASK;
                let tmp = perm_table[k];
                perm_table[k] = perm_table[i];
                perm_table[i] = tmp;
                perm_table[k + SIZE] = tmp;
            }

            noise_map[y * SIZE + x] = eval(x as f32 * freq, y as f32 * freq, &perm_table, &r);
        }
    }

    noise_map

}

fn eval(x: f32, y: f32, permTable: &Vec<usize>, r: &Vec<f32>) -> f32 {
    let xi = x as usize;
    let yi = y as usize;

    let tx = x - xi as f32;
    let ty = y - yi as f32;

    let rx0 = xi & SIZE_MASK;
    let rx1 = (rx0 + 1) & SIZE_MASK;
    let ry0 = yi & SIZE_MASK;
    let ry1 = (ry0 + 1) & SIZE_MASK;

    let c00 = r[permTable[permTable[rx0] + ry0]];
    let c10 = r[permTable[permTable[rx1] + ry0]];
    let c01 = r[permTable[permTable[rx0] + ry1]];
    let c11 = r[permTable[permTable[rx1] + ry1]];

    let sx = smoothstep(tx);
    let sy = smoothstep(ty);

    let nx0 = lerp(c00, c10, sx);
    let nx1 = lerp(c01, c11, sx);

    lerp(nx0, nx1, sy)
}

pub fn make_map() -> Vec<render::Triangle> {
    let squares = mk_cylinder();
    triangleify(&squares)
}
