model=microsoft/Phi-3-medium-4k-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
NUM_GPUS=8
token="<paste_your_huggingface_hub_token_here>"


docker run --gpus all --shm-size 20g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model --revision refs/pr/68 #--sharded true --num-shard $NUM_GPUS #--quantize bitsandbytes