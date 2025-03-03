model=path/to/saved/merged/model

volume=$PWD/training # share a volume with the Docker container to avoid downloading weights every run
quantize=bitsandbytes
NUM_GPUS=8

docker run --gpus all --shm-size 1g -p 8080:80 -v /home:/home \
            -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.1 \
            --model-id $model  --trust-remote-code