# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:29
# @author  : Mo
# @function: config of llama


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # default=4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4  # default=3e-4  # the Karpathy constant
# LEARNING_RATE = 3e-4  # default=3e-4  # the Karpathy constant
EPOCHS = 3  # default=3  # we don't always need 3 tbh
# LORA_DROPOUT = 0.1
# LORA_ALPHA = 32
# LORA_R = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 16
LORA_R = 8
SAVE_STEPS = 382
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 256 - 2  # default=128 - 2
MAX_LENGTH_A = 256 - 2  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
TARGET_MODULES = ["c_attn",
                  "c_proj",
                  ]

PATH_MODEL_PRETRAIN = ""
REPO_ID = "Qwen/Qwen-7B-Chat"
PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN if PATH_MODEL_PRETRAIN else REPO_ID
DATA_PATH = "../dataset/alpaca_gpt4_data_zh.json"
MODEL_SAVE_DIR = "model_sft"

IS_PARALLELIZABLE = True
MODEL_PARALLEL = True
USE_CACHE = False
CUDA_VISIBLE_DEVICES = "0"
USE_TORCH = "1"
CPU_NUMS = "9"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True


"""
'transformer.wte.weight'
'transformer.h.0.ln_1.weight'
'transformer.h.0.attn.c_attn.weight'
'transformer.h.0.attn.c_attn.bias'
'transformer.h.0.attn.c_proj.weight'
'transformer.h.0.ln_2.weight'
'transformer.h.0.mlp.w1.weight'
'transformer.h.0.mlp.w2.weight'
'transformer.h.0.mlp.c_proj.weight'
'transformer.h.1.ln_1.weight'
'transformer.h.1.attn.c_attn.weight'
'transformer.h.1.attn.c_attn.bias'
'transformer.h.1.attn.c_proj.weight'
'transformer.h.1.ln_2.weight'
'transformer.h.1.mlp.w1.weight'
'transformer.h.1.mlp.w2.weight'
'transformer.h.1.mlp.c_proj.weight'
'transformer.ln_f.weight'
'lm_head.weight'
"""
