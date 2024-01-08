import base64 
import json 

import numpy as np 
import cv2 

import torch 
import torch.nn.functional as F 

import triton_python_backend_utils as pb_utils

def score_to_logit(score):
    score = torch.tensor(score)
    logit = F.softmax(score, dim=1)
    class_id = logit.argmax().item()
    prob = logit[0][class_id].item() * 100.
    return prob, class_id

def wrap_json(prob,
              pred):
    """
    1. tensor logit to list 
    2. make json output 
    """
    obj = {
        "prob": prob,
        "pred": pred
    }
    return json.dumps(obj, ensure_ascii=False)

class TritonPythonModel:
    """
    post processing main logic
    """
    def initialize(self, args):
        self.class2label = {}
        with open("/tmp/imagenet_label.txt", "r") as f:
            labels = f.readlines()
        for i, label in enumerate(labels):
            self.class2label[i] = label.replace('\n', '')

    def execute(self, requests):
        responses = [] 
        for request in requests: 
            predict_score = pb_utils.get_input_tensor_by_name(request, "SCORES").as_numpy()
            
            prob, class_id = score_to_logit(predict_score)
            pred = self.class2label[class_id]
            response = np.array(wrap_json(prob, pred), dtype=np.object_)
            response = pb_utils.Tensor("RESULT", response)
            response = pb_utils.InferenceResponse(output_tensors=[response])
            responses.append(response)
        return responses


