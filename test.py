# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64

model_inputs = {'prompt': 'A dog'}
model_inputs = {'image': "img_url"}

res = requests.post('http://localhost:8000/', json = model_inputs)

json_data = res.json()

fh = open("pointcloud.ply", "wb")
fh.write(base64.b64decode(json_data["ply"]))
fh.close()

fh = open("mesh.glb", "wb")
fh.write(base64.b64decode(json_data["glb"]))
fh.close()
