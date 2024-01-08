import base64
import os
import requests
import json
import numpy as np

from pytictoc import TicToc

# TicToc 클래스 생성
t = TicToc()

IP = "128.0.0.1"  ## use your ip


def inference(image_data, url="localhost", port="8005"):
    url = f"http://{url}:{port}/v2/models/ensemble/infer"
    data = {
        "inputs": [
            {
                "name": "raw_image",
                "shape": [len(image_data)],
                "datatype": "BYTES",
                "data": image_data,
            }
        ]
    }
    headers = {"content-type": "application/json"}

    t.tic()
    response = requests.post(
        url, headers=headers, data=json.dumps(data, ensure_ascii=False)
    )
    tm = t.tocvalue()
    return response.text, tm


def read_image_data(im_paths):
    encode_ims = []
    for p in im_paths:
        if not os.path.exists(p):
            continue
        image = open(p, "rb")
        im_encode = base64.b64encode(image.read()).decode("ascii")
        encode_ims.append(im_encode)
    return encode_ims


if __name__ == "__main__":
    im_path = ["/home1/irteamsu/workspace/da2so/triton_server_example/goldfish.jpg"]
    images = read_image_data(im_path)
    response, tm = inference(images, IP)
    print("Inference Time: %f" % tm)
    response_json = json.loads(response)
    response_data = json.loads(response_json["outputs"][0]["data"][0])
    prob = response_data["prob"]
    pred = response_data["pred"]
    print(f'Pred: {pred} (prob: {prob:.3f})')