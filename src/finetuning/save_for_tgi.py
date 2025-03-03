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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os
from autotrain.tools.merge_adapter import merge_llm_adapter

models_path = "<path_to_fine_tuning_folder>"

filenames = ["phi3-14b-ft-context-bf16-ga2-int4", "phi3-14b-ft-param-bf16-ga2-int4"]

for adapter_model_id in tqdm(filenames):
    model_id = "microsoft/Phi-3-medium-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    merge_llm_adapter(model_id, models_path+adapter_model_id, tokenizer, output_folder="<path_to_chechkpoint_storing_directory>"+adapter_model_id)

