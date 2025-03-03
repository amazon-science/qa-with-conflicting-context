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
from text_generation import Client
import pickle

def tgi_call(text):
    out1 = client.generate(text, max_new_tokens=128, temperature=0.1, best_of=1, return_full_text=True, seed=42, do_sample=False).generated_text
    return out1.strip()

with open("ambig_qa_answers_expert_eval.pickle", "rb") as file:
    lst= pickle.load(file)


data={}
data["prompt"]=[]
for each in lst:
    question = each["question"]
    result = each["result"]
    contexts = []
    sources = []

    # prompt = "Answer the question concisely. \n\n"
    prompt = "Answer the question based on the contexts below. Keep the answer short. \n\n"
    # prompt = "Explain which answer do you think is correct based on the context below and write down your final answer. Keep the answer short. \n\n"

    for i in range(len(result)):
        if "snippet" in result[i]:
            snippet = result[i]["snippet"]
            contexts.append(snippet)

        else:
            continue
        if "//" in result[i]["formattedUrl"]:
            source = result[i]["formattedUrl"].split("//")[1].split("/")[0]
        else:
            source = result[i]["formattedUrl"].split("/")[0]
        sources.append(source)


    for i in range(len(contexts)):
        prompt += "Context" + str(i + 1) + " from "+ sources[i] +": "+ contexts[i] + "\n\n"

    prompt += "Question: " + question + "\n\n"

    prompt += "Answer: "
    data["prompt"].append(prompt)
    print(prompt)

client = Client("http://127.0.0.1:8080", timeout=300)

# t_start = time.time()
with multiprocess.Pool(127) as p:
    outputs = tqdm(p.imap(tgi_call, data['prompt']), total=len(data["prompt"]))
    outputs = list(outputs)

with open("path/to/saved/generations", "wb") as file:
    pickle.dump(outputs, file)