  # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  # Licensed under the Apache License, Version 2.0 (the "License").
  # You may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  
  #     http://www.apache.org/licenses/LICENSE-2.0
  
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

import multiprocess
from tqdm import tqdm
import pickle
from huggingface_hub import InferenceClient
import json
import pandas as pd



with open("../../data/ConflictQA_Dataset.json", "r") as file:
    conflictqa = json.load(file)


test= []
for each in conflictqa:
    if "test" not in each["batch"]:
        continue
    test.append(each)



data=[]
answers =[]


for each in test:
    message = []
    dic={}
    dic["role"] = "system"
    dic["content"] = "Given the following contexts, provide a direct, concise answer to the question. Make the answer as short as possible."

    message.append(dic)
    prompt = ""
    question = each["question"]
    question = question[0].upper() + question[1:]

    contexts = each["contexts"]
    sources = each["sources"]


    for i in range(len(contexts)):
        prompt += "Context" + str(i + 1) + " from " + sources[i] + ": " + contexts[i] + "\n\n"


    prompt += question


    dic = {}
    dic["role"] = "user"
    dic["content"] = prompt
    message.append(dic)

    data.append(message)



client = InferenceClient(model="http://127.0.0.1:8080")



def tgi_call(text):
    out1 = client.chat_completion(text, seed=42, temperature=0, model="http://127.0.0.1:8080")
    return out1.choices[0].message.content

with multiprocess.Pool(127) as p:
    outputs = tqdm(p.imap(tgi_call, data), total=len(data))
    outputs = list(outputs)

with open("../../data/2024-new-data/generations/phi3-zeroshot-context-exp-conflictqa.pickle", "wb") as file:
    pickle.dump(outputs, file)