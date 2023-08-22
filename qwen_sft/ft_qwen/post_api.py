# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: fastapi-post接口


import traceback
import logging
import random
import time
import json
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from qwen_sft.ft_qwen.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch

from pydantic import BaseModel
from fastapi import FastAPI
import time

# from transformers import LlamaForCausalLM, LlamaModel
# from transformers import LlamaTokenizer, LlamaConfig
from qwen_sft.models.qwen.modeling_qwen import QWenLMHeadModel as LLMModel
from qwen_sft.models.qwen.tokenization_qwen import QWenTokenizer as LLMTokenizer
from qwen_sft.models.qwen.configuration_qwen import QWenConfig as LLMConfig
from qwen_sft.ft_qwen.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from qwen_sft.ft_qwen.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from qwen_sft.ft_qwen.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from qwen_sft.ft_qwen.config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from qwen_sft.ft_qwen.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from qwen_sft.ft_qwen.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from qwen_sft.ft_qwen.config import USE_CUDA


app = FastAPI()  # 日志文件名,为启动时的日期, 全局日志格式
logger_level = logging.INFO
logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] "
                           "- %(levelname)s: %(message)s",
                    level=logger_level)
logger = logging.getLogger("ft-llama")
console = logging.StreamHandler()
console.setLevel(logger_level)
logger.addHandler(console)


# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
device_map = "auto"
# USE_CUDA = True
print(device_map)
print(ddp)


def load_model_state(model, model_save_dir="./", model_name="pytorch_model.bin", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)
        state_dict = torch.load(path_model, map_location=torch.device(device))
        # print(state_dict.keys())
        state_dict = {"base_model.model." + k.replace("_orig_mod.", "")
                      .replace(".lora_A.weight", ".lora_A.default.weight")
                      .replace(".lora_B.weight", ".lora_B.default.weight")
                      : v for k, v in state_dict.items()}
        print(state_dict.keys())
        print("#" * 128)
        ### 排查不存在model.keys的 state_dict.key
        name_dict = {name: 0 for name, param in model.named_parameters()}
        print(name_dict.keys())
        print("#" * 128)
        for state_dict_key in state_dict.keys():
            if state_dict_key not in name_dict:
                print("{} is not exist!".format(state_dict_key))
        model.load_state_dict(state_dict, strict=False)
        # model.to(device)
        print("******model loaded success******")
        print("self.device: {}".format(device))
    except Exception as e:
        print(str(e))
        raise Exception("******load model error******")
    return model
def save_model_state(model, config=None, model_save_dir="./", model_name="pytorch_model.bin"):
    """  仅保存模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if config:
        config.save_pretrained(model_save_dir)
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    # torch.save(model.state_dict(), path_model)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def txt_read(path, encode_type="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encode_type, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines
def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json
def generate_prompt_offical(data_point, is_logger=False):
    """   官方 prompt """
    from qwen_sft.models.qwen.qwen_generation_utils import make_context
    system = "You are a helpful assistant."
    query = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    raw_text, context_tokens = make_context(
                tokenizer,
                query.strip(),
                history=[],
                system=system,
                max_window_size=6144
    )
    out = {"input_ids": context_tokens}
    return out
def generate_prompt_train(data_point, is_logger=False):
    """ 直接encode，最后强制补上(不走新增了)，
    稍微有所不同，推理的text_1需要加上assistant\n, 与训练稍有不同,
    并且终止条件为<|im_start|>、<|im_end|>、<|endoftext|>三个
    """
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"
    system_str = "You are a helpful assistant."
    prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(system_str)
    text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    text_out = data_point.get("output", "")
    prompt_text_1 = prompt_system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_text_2 = "{}<|im_end|><|endoftext|>"
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out)
    # end with <|im_end|><|endoftext|>
    x = tokenizer.encode(text_1)
    y = tokenizer.encode(text_2)
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    if not x:
        x = [ID_SOP, ID_EOP, ID_BR, ID_SOP]
    x[-3:] = [ID_EOP, ID_BR, ID_SOP]
    if not y:
        y = [ID_EOP, ID_EOS]
    y[-2:] = [ID_EOP, ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
    return out
def generate_prompt(data_point, is_logger=False):
    """ 直接encode，最后强制补上(不走新增了)，
    稍微有所不同，推理的text_1需要加上assistant\n, 与训练稍有不同,
    并且终止条件为<|im_start|>、<|im_end|>、<|endoftext|>三个
    """
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"
    system_str = "You are a helpful assistant."
    prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(system_str)
    text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    text_out = data_point.get("output", "")
    prompt_text_1 = prompt_system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_text_2 = "{}<|im_end|><|endoftext|>"
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out)
    # end with <|im_end|><|endoftext|>
    x = tokenizer.encode(text_1)
    y = tokenizer.encode(text_2)
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    if not x:
        x = [ID_SOP, ID_EOP, ID_BR, ID_SOP, ID_ASSISTANT, ID_BR]
    x[-5:] = [ID_EOP, ID_BR, ID_SOP, ID_ASSISTANT, ID_BR]
    if not y:
        y = [ID_EOP, ID_EOS]
    y[-2:] = [ID_EOP, ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
    return out

model = LLMModel.from_pretrained(PATH_MODEL_PRETRAIN)
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=False,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["ln_1",
                          "ln_2",
                          "ln_f",
                          ],
        )
config = LoraConfig(target_modules=TARGET_MODULES,
                    lora_dropout=LORA_DROPOUT,
                    lora_alpha=LORA_ALPHA,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=LORA_R,
                    )
model = get_peft_model(model, config)
model = load_model_state(model=model, model_save_dir=MODEL_SAVE_DIR)
if USE_CUDA:
    model = model.cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, use_print_data=True)
# print_named_parameters(model)

tokenizer = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN, add_eos_token=True)
ID_PAD = 151643
ID_EOS = 151643  # endoftext
ID_SOP = 151644  # start
ID_EOP = 151645  # end
ID_BR = 198  # "\n"
ID_ASSISTANT = 77091  # assistantr
tokenizer.pad_token_id = ID_PAD
tokenizer.eos_token_id = ID_EOS
### 推理中的assistant\n在问句中
# STOP_WORDS_IDS = [[ID_EOS]]
STOP_WORDS_IDS = [[ID_SOP], [ID_EOP], [ID_EOS]]
# tokenizer.padding_side = "right"  # NO attention-mask
tokenizer.padding_side = "left"
print(ID_PAD)
print(ID_EOS)
print(ID_SOP)
print(ID_EOP)
print(ID_BR)
print(ID_ASSISTANT)
# ENDOFTEXT = "<|endoftext|>"
# IMSTART = "<|im_start|>"
# IMEND = "<|im_end|>"


def predict(data_point, generation_config):
    """  推理  """
    prompt_dict = generate_prompt(data_point)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(**generation_config)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stop_words_ids=STOP_WORDS_IDS,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(input_ids)
    print(s)
    print(output)
    output = output.strip().replace("<|im_end|><|endoftext|>", "").split("<|im_start|>assistant\n")[-1]
    return output
class Item(BaseModel):
    instruction: str = ""
    text: str = "1+1="
    penalty_alpha: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.8  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.8  # 0.75
    top_k: int = 50


@app.post("/nlg/text_generate")
def text_generate(request_data: Item):
    instruction = request_data.instruction
    text = request_data.text
    penalty_alpha = request_data.penalty_alpha
    max_new_tokens = request_data.max_new_tokens
    temperature = request_data.temperature
    do_sample = request_data.do_sample
    num_beams = request_data.num_beams
    top_p = request_data.top_p
    top_k = request_data.top_k

    generation_dict = vars(request_data)
    print(generation_dict)
    generation_dict.pop("max_new_tokens")
    generation_dict.pop("instruction")
    generation_dict.pop("text")
    data_point = {"instruction": instruction, "input": text, "output": ""}
    generation_config = {"temperature": temperature,
                         "top_p": top_p,
                         "top_k": top_k,
                         "num_beams": num_beams,
                         "do_sample": do_sample,
                         "penalty_alpha": penalty_alpha,
                         "max_new_tokens": max_new_tokens,
                         "pad_token_id": ID_PAD,
                         "eos_token_id": ID_EOS,
                         }
    try:  # 数据预处理, 模型预测
        response = predict(data_point, generation_config)
    except Exception as e:
        logger.info(traceback.print_exc())
        response = "<endoftext>"
    return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8044,
                workers=1)


"""
# nohup python post_api.py > tc.post_api.py.log 2>&1 &
# tail -n 1000  -f tc.post_api.py.log
# |myz|

可以在浏览器生成界面直接访问: http://localhost:8036/docs
"""