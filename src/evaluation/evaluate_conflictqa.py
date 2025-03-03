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


import numpy as np
import pickle
import json

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


with open("../../data/ConflictQA_Dataset_5_batches.json", "r") as file:
    dic = json.load(file)
print(dic[0].keys())


with open("../../data/2024-new-data/generations/gpt4o-fewshot-context-conflictqa5.pickle", "rb") as file:
    preds = pickle.load(file)



em_total = []
f1_total = []
em_c = []
em_nc = []
f1_c = []
f1_nc = []
hallucinate_exp = 0
hallucinate_context = 0


new_dic = []
for each in dic:
    if "test" not in each["batch"]:
        continue
    new_dic.append(each)

dic = new_dic
print(len(dic))
reasons = []

for i in range(len(dic)):
    answer = dic[i]["ambigqa_answer"]


    if "Answer: " in preds[i]:
        pred = preds[i].split("Answer: ")[1].split("\n\n")[0]
    else:
        pred = preds[i].split("\n\n")[0]
    if "<|end|>" in pred:
        pred = pred.replace("<|end|>","")


    max_em = -1
    max_em_wrong = -1

    for ref in answer:
        score = compute_exact_match(pred, ref)
        if score>max_em:
            max_em=score


    max_f1 = -1
    max_f1_wrong = -1
    for ref in answer:
        score = compute_f1(pred, ref)

        if score>max_f1:
            max_f1 = score



    if dic[i]["secondAnswerExist"] == "A":
        em_c.append(max_em)
        f1_c.append(max_f1)
    else:
        em_nc.append(max_em)
        f1_nc.append(max_f1)


    answer = [normalize_text(each) for each in answer]

    if dic[i]["secondAnswerExist"] == "A":
        reasons.append(dic[i]["reasons"])
        context = ". ".join(dic[i]["contexts"]).lower()
        if pred not in context:

            temp =0
            for each in answer:
                if each in pred:
                    temp += 1

            hallucinate_exp += max(temp,0)



    em_total.append(max_em)
    f1_total.append(max_f1)

print(len(em_total))

# print(pred)
print("EM Conflict", np.round(np.mean(em_c)*100,2))
print("EM Non Conflict", np.round(np.mean(em_nc)*100, 2))
print("EM:", np.round(np.mean(em_total)*100, 2))

print("F1 Conflict", np.round(np.mean(f1_c)*100, 2))
print("F1 Non Conflict", np.round(np.mean(f1_nc)*100,2))
print("F1:", np.round(np.mean(f1_total)*100, 2))