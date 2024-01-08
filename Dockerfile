FROM nvcr.io/nvidia/tritonserver:23.03-py3
COPY ./models /models
COPY ./imagenet_label.txt /tmp/imagenet_label.txt
RUN apt-get update
RUN pip3 install numpy opencv-python torch albumentations 
RUN apt install -y libgl1-mesa-glx
ENTRYPOINT ["tritonserver", "--model-repository=/models"]