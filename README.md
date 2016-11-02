# My own 3d graphics drawing API

Runs a rust websocket server (rust code in src). The user interface is javascript that decompresses the images from the 
rust server and sends back keyboard input.

## Not working:
- Frustum culling does not really work, yet
- It's way too slow, but could be sped up
- No shaders or textures or anything fancy like that
