extern crate obj_parser;
extern crate websocket;
//extern crate bresenham;
extern crate nalgebra as na;
extern crate flate2;
extern crate rand;

use websocket::{Server, Message, Sender, Receiver};
use websocket::message::Type;
use websocket::header::WebSocketProtocol;
use obj_parser::object::{parse_file_to_object, Object};
use na::{Vector3};

use std::{thread, time};
use std::io::prelude::*;
use std::sync::mpsc;
use std::str;

mod render;
mod draw;
mod noise;

type ScreenBuf = Vec<u8>;
type InputBuf = Vec<u8>;

fn main() {
    let server = Server::bind("127.0.0.1:2794").unwrap();

    // let finger = parse_file_to_object("finger.obj").unwrap();
    let head = parse_file_to_object("african_head.obj").unwrap();
    let tris = get_triangles(head);
    // let tris = noise::make_map();
    
    let (tx, rx): (mpsc::Sender<ScreenBuf>, mpsc::Receiver<ScreenBuf>) = mpsc::channel();
    let (inp_tx, inp_rx): (mpsc::Sender<InputBuf>, mpsc::Receiver<InputBuf>) = mpsc::channel();

    thread::spawn(move || {
        render::run(tx, inp_rx, &tris);
    });

    for connection in server {
        //let mut screen_buf = screen_buf.clone();

        // Spawn a new thread for each connection.
        let request = connection.unwrap().read_request().unwrap(); // Get the request
        let headers = request.headers.clone(); // Keep the headers so we can check them

        request.validate().unwrap(); // Validate the request

        let mut response = request.accept(); // Form a response

        if let Some(&WebSocketProtocol(ref protocols)) = headers.get() {
            if protocols.contains(&("rust-websocket".to_string())) {
                // We have a protocol we want to use
                response.headers.set(WebSocketProtocol(vec!["rust-websocket".to_string()]));
            }
        }

        let mut client = response.send().unwrap(); // Send the response

        let ip = client.get_mut_sender()
            .get_mut()
            .peer_addr()
            .unwrap();

        println!("Connection from {}", ip);

        let (mut sender, mut receiver) = client.split();

        // lol
        thread::spawn(move || {
            for message in receiver.incoming_messages() {

                let message: Message = message.unwrap();

                match message.opcode {
                    // _ => match str::from_utf8(&message.payload) {
                    //     Ok(s) => println!("Got: {}", s),
                    //     Err(e) => println!("Err: {}", e),
                    _ => inp_tx.send(message.payload.into_owned()).unwrap(),
                }
            }
        });

        loop {

            if let Ok(screen) = rx.try_recv() {
                let message: Message = Message::binary(screen);

                sender.send_message(&message).unwrap();
            }

            thread::sleep(time::Duration::from_millis(16));

        }

    }
}

pub fn get_triangles(obj: Object) -> Vec<render::Triangle> {
    let mut tris = Vec::new();
    for face in obj.faces {
        let mut tri = Vec::new();
        for idx in face.vertex_indices {
            let index = idx.vertex_index - 1;
            let vertex = &obj.vertices[index];
            let v = Vector3::new(vertex.x,
                                 vertex.y,
                                 vertex.z,
                                 );
            tri.push(v);
        }
        tris.push(tri);
    }
    tris
}

