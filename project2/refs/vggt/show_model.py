import viser

glb_path = "/home/cxx/HWs/CS290U/project2/refs/vggt/outs/vggt_data1.glb"
server = viser.ViserServer(host="0.0.0.0", port=8081)

# 1️⃣ 以二进制方式读取 .glb 文件
with open(glb_path, "rb") as f:
    glb_bytes = f.read()

# 2️⃣ 传给 add_glb()
server.scene.add_glb("my_model", glb_bytes, scale=1.0)

print("✅ Model loaded successfully! Open http://localhost:8081")
while True:
    pass
