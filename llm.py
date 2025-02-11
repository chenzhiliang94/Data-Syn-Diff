import torch_influence
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_warn_always(False)
from copy import deepcopy
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import lm_eval
from lm_eval.models.huggingface import HFLM

import datasets
import os
import sys

import transformers
from datasets import load_dataset, concatenate_datasets
import gc

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)

from transformers import LlamaForCausalLM, LlamaTokenizer

def get_tokenizer_and_model(model_id = "meta-llama/Llama-3.1-8B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # adjust tokenizer (from alpaca repo, MIGHT NOT BE NEEDED IDK)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    model = model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto'
    )

    return tokenizer, model

def load_data(data_domain):
        # Load the dataset
    print(data_domain)
    if data_domain == "headqa_en":
        data_domain = "headqa"
    if data_domain == "wikitext":
        dataset = datasets.load_dataset(data_domain, "wikitext-2-v1", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "triviaqa":
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "pubmedqa":
        dataset = datasets.load_dataset("bigbio/pubmed_qa", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "truthfulqa_gen":
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", cache_dir = "./datasets")
        train_dataset = dataset["validation"]
        val_dataset = dataset["validation"]
    elif data_domain == "commonsense_qa":
        dataset = datasets.load_dataset("tau/commonsense_qa", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "hellaswag":
        dataset = datasets.load_dataset("DatologyAI/hellaswag", cache_dir = "./datasets", trust_remote_code=True)
        train_dataset = dataset["eval"]
        val_dataset = dataset["eval"]
    elif data_domain == "sciq":
        dataset = datasets.load_dataset("allenai/sciq", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    elif data_domain == "squadv2":
        dataset = datasets.load_dataset("rajpurkar/squad_v2" , cache_dir = "./datasets")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "headqa":
        dataset = datasets.load_dataset("dvilares/head_qa", "en", cache_dir = "./datasets", trust_remote_code=True)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    elif data_domain == "hellaswag":
        dataset = datasets.load_dataset("DatologyAI/hellaswag", cache_dir = "./datasets", trust_remote_code=True)
        train_dataset = dataset["eval"]
        val_dataset = dataset["eval"]
    else:
        assert False, "data_domain not valid, pls check"
    return train_dataset, val_dataset