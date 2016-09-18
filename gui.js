var index = 4;

var pressed = {
    37: false,      // left
    38: false,      // up
    39: false,      // right
    40: false       // down
};

$(document).ready(function() {
    var socket = new WebSocket("ws://127.0.0.1:2794");
    socket.binaryType = "arraybuffer";

    var canvas = $("#canvas")[0];
    $("#canvas").css("cursor", "none");
    
    var ctx = canvas.getContext("2d");

    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;

    var keycodes = new Uint8Array(100);
    var mousePos = {};

    // var lastCalledTime;
    // var fps;

    var image = ctx.createImageData(canvas.width, canvas.height);
    var change = true;

    $(document).mousemove(function(e) {
        change = true;
        mousePos = {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
        // console.log((keycodes[0] & 0xFF) | ((keycodes[1] & 0xFF) << 8));
        // console.log(keycodes);
    });

    $(document).keydown(function(e) {
        change = true;
        pressed[e.which] = true;
        for (var code in pressed) {
          if (pressed.hasOwnProperty(code) && pressed[code]) {
              keycodes[index] = code;
              index += 1;
          }
        }
        e.preventDefault();
    });

    $(document).keyup(function(e) {
        change = true;
        pressed[e.which] = false;
        e.preventDefault();
    });

    // fps counter slows this down, a lot. no idea how many frames though
    socket.onmessage = function (e) {
        var bytes = new Uint8Array(e.data);
        var out = pako.inflate(bytes);
        for (var i = 0; i < image.data.length; i += 4) {
            image.data[i + 0] = out[i + 0];
            image.data[i + 1] = out[i + 1];
            image.data[i + 2] = out[i + 2];
            image.data[i + 3] = 255;
        }
        ctx.putImageData(image, 0, 0);
        keycodes[0] = mousePos.x & 0xFF;
        keycodes[1] = (mousePos.x >> 8) & 0xFF;
        keycodes[2] = mousePos.y & 0xFF;
        keycodes[3] = (mousePos.y >> 8) & 0xFF;

        if (change) {
            var codes = keycodes.slice(0, index).buffer;
            socket.send(codes);
        }

        change = false;
        index = 4;

//        if (!lastCalledTime) {
//            lastCalledTime = Date.now();
//            fps = 0;
//            return;
//        }
//        delta = (Date.now() - lastCalledTime) / 1000;
//        lastCalledTime = Date.now();
//        fps = 1 / delta;
//
//        ctx.fillStyle = "white";
//        ctx.font = "12px mono";
//        ctx.fillText(fps.toFixed(2) + " fps", 10, 20)
    }
});
