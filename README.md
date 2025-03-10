This repository contains the data and the sources used in the following paper. Please consider citing it if you use this repository.

```
@inproceedings{
liu2025open,
title={Open Domain Question Answering with Conflicting Contexts},
author={Siyi Liu and Qiang Ning and Kishaloy Halder and Wei Xiao and Zheng Qi and Phu Mon Htut and Yi Zhang and Neha Anna John and Bonan Min and Yassine Benajiba and Dan Roth},
booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
year={2025},
}
```

Steps to reproduce all the artifacts and results of the project:

    Dependencies:
        Install all dependencies by running pip install -r requirements.txt

    Finetuning:
        1. Run src/finetuning/train.sh to finetune a model you'd like to use with PEFT. Make sure you change --model to a model name you'd like to use on Huggingface, and --data_path to the path of your prepared training data. This will save only an adapter model and not the whole base model.
        2. Run src/finetuning/save_for_tgi.py to merge the adapter model you saved in the previous step and the base model together and save the merged model. This is for the purpose of faster inference with text generation inference (TGI) as TGI only takes a full model as input.

    Generation:
        1. Run src/generation/start_server_local.sh to start a TGI server with the model you just finetuned, merged, and saved locally.
        2. Run src/generation/start_server_huggingface.sh if you want to start a TGI server with a model on huggingface, instead of a model you saved.
        3. Run src/generation/text_generation_inference.py to inference with the server you just started. The input data of this file can either be the expert evaluation set (ambig_qa_answers_expert_eval.pickle) or the NQ_open test set(nq_open_test_1000.pickle).

    Evaluation:
        1. Run src/evaluation/evaluate_expert.py to print the evaluation results of the expert evaluation set.
        2. Run src/evaluation/evaluate_nq_test.py to print the evaluation results of the NQ-open test set.

Specific descriptions of each directory and file:

data/ConflictQA_Dataset.json: is the full corpus that we collected

src/evaluation/: contains scripts that evaluate the generation of a model with the reference answer
    - evaluate_expert.py: prints the EM and F1 scores of our expert evaluation set
    - evaluate_nq_test.py: prints the EM and F1 scores of our 1,000 random NQ-Open test set.

src/finetuning/: contains scripts used for finetuning
    - save_for_tgi.py: contains the code used to merge the adapter model saved by train.sh with the base model and save the merged model.
    - train.sh: finetunes and saves an adapter model using the autotrain-advanced package.

src/generation/: contains the code used for model inference/generation
    - start_server_huggingface.sh: starts a server in docker with the name of a model from huggingface.
    - start_server_local.sh: starts a server in docker using a model saved locally.
    - text_generation_inference.py: generates given the prompts when there's a server started
