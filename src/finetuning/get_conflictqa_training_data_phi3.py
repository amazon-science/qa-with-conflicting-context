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

import json
import pandas as pd
import ast
import csv

with open("../../data/ConflictQA_Dataset.json") as file:
    data = json.load(file)

print(data[0])
training_data = []
testing_data = []


ID_match = {"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "H":8, "I":9, "J":10}


system_prompt = "You are a knowledgeable, efficient, and direct AI assistant. Given the following contexts, directely answer the question with as few words as possible. Do not use more than five words.\n\n"

for j, line in enumerate(data):

    context = ""
    for i in range(len(line["contexts"])):
        context += "Context" + str(i+1) + " from " + line["sources"][i] + ": " + line["contexts"][i] + "\n\n"

    if line["correctAnswer"] not in [line["firstAnswer"], line["secondAnswer"],line["thirdAnswer"]]:
        if "only" in line["correctAnswer"] or "most" in line["correctAnswer"] or "fully" in line["correctAnswer"] \
        or "can" in line["correctAnswer"]:
            continue
    question = line["question"]
    if question[-1]!="?":
        question = question + "?"

    question = question[0].upper() + question[1:]


    text = "<|user|>\n" + system_prompt + context + question + "<|end|>\n" + "<|assistant|>\n" + \
             "Answer: "+ str(line["correctAnswer"]) + "<|end|>"



    if "test" in line["batch"]:
        testing_data.append(text)
    else:
        training_data.append(text)




with open("conflictqa_train_phi3_context.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["test"])
    for row in training_data:
        writer.writerow([row])
