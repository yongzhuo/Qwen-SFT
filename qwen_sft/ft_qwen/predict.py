# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 推理


import random
import time
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from qwen_sft.ft_qwen.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from peft import (get_peft_model, LoraConfig)
from transformers import GenerationConfig
import torch

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
def print_named_parameters(model, use_print_data=True):
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


def predict(data_dict):
    """  推理  """
    prompt_dict = generate_prompt(data_dict)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(
        # temperature=0.95,
        # top_p=0.75,
        temperature=0.8,
        top_p=0.8,
        top_k=50,
        num_beams=1,
        do_sample=True,
        max_new_tokens=256,
        # penalty_alpha=1.5,
        pad_token_id=ID_PAD,
        eos_token_id=ID_EOS,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stop_words_ids=STOP_WORDS_IDS,
            # stop_words_ids=[[ID_EOS]],
            return_dict_in_generate=True,
            # return_dict_in_generate=True,
            # output_scores=True,
            # max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(data_dict)
    print(input_ids)
    print(output)
    # output = output.split("答：")[-1]
    return output



if __name__ == '__main__':
    data_dict = {"instruction": "解释为什么下面的分数等于 1/4",
                 "input": "解释为什么下面的分数等于 1/4，4/16",
                 "output": "分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。"
                 }
    res = predict(data_dict)
    print(res)
    while True:
        time_start = time.time()
        history = []
        print("请输入:")
        ques = input()
        print("请稍等...")
        try:
            if ques.strip().upper() == "CLEAR":
                history = []
                print("clear ok")
                continue
            else:
                print("#" * 128)
                ques_dict = {"instruction": ques, "input": "", "output": ""}
                res = predict(ques_dict)
                print(res)
                print("###mid###" * 32)
                response, history = model.chat(tokenizer, ques, history=history)
                print(response)
                print("#" * 128)
        except Exception as e:
            print(str(e))
        print(time.time() - time_start)

"""
python predict.py

### 打印了2次并与chat对比
trainable params: 0 || all params: 7731482624 || trainable%: 0.0
151643
151643
151644
151645
{'instruction': '解释为什么下面的分数等于 1/4', 'input': '解释为什么下面的分数等于 1/4，4/16', 'output': '分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。'}
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 104136, 100678, 100431,   9370,
         103190, 107106,    220,     16,     14,     19,    197, 104136, 100678,
         100431,   9370, 103190, 107106,    220,     16,     14,     19,   3837,
             19,     14,     16,     21, 151645,    198, 151644]])
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
解释为什么下面的分数等于 1/4    解释为什么下面的分数等于 1/4，4/16<|im_end|>
<|im_start|>assistant
当把一个分数的分子和分母都除以相同的数时，分数大小不变，这个数就叫做这个分数的最小公倍数。4/16的分子和分母的最大公约数是4，所以4/16的分数可以写成1/4的形式。<|im_end|><|endoftext|>
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
解释为什么下面的分数等于 1/4    解释为什么下面的分数等于 1/4，4/16<|im_end|>
<|im_start|>assistant
当把一个分数的分子和分母都除以相同的数时，分数大小不变，这个数就叫做这个分数的最小公倍数。4/16的分子和分母的最大公约数是4，所以4/16的分数可以写成1/4的形式。<|im_end|><|endoftext|>
请输入:
类型#上衣*风格#街头*图案#创意*衣样式#卫衣
请稍等...
################################################################################################################################
{'instruction': '类型#上衣*风格#街头*图案#创意*衣样式#卫衣', 'input': '', 'output': ''}
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2,  17447,  99741,
              9, 104040,      2, 108094,      9, 108108,      2, 102343,      9,
          99741, 112453,      2,  99417,  99741, 151645,    198, 151644]])
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#上衣*风格#街头*图案#创意*衣样式#卫衣<|im_end|>
<|im_start|>assistant
卫衣是很多时尚潮人喜爱的单品，而这款卫衣最大的亮点就在于衣身的logo设计，不仅起到了装饰的作用，而且彰显出品牌的独特创意。再加上oversize的版型，街头范儿十足。<|im_end|><|endoftext|>
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#上衣*风格#街头*图案#创意*衣样式#卫衣<|im_end|>
<|im_start|>assistant
卫衣是很多时尚潮人喜爱的单品，而这款卫衣最大的亮点就在于衣身的logo设计，不仅起到了装饰的作用，而且彰显出品牌的独特创意。再加上oversize的版型，街头范儿十足。<|im_end|><|endoftext|>
###mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid###
################################################################################################################################
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#上衣*风格#街头*图案#创意*衣样式#卫衣<|im_end|>
<|im_start|>assistant

[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 31905, 2, 17447, 99741, 9, 104040, 2, 108094, 9, 108108, 2, 102343, 9, 99741, 112453, 2, 99417, 99741, 151645, 198, 151644, 77091, 198]
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2,  17447,  99741,
              9, 104040,      2, 108094,      9, 108108,      2, 102343,      9,
          99741, 112453,      2,  99417,  99741, 151645,    198, 151644,  77091,
            198,  99417,  99741,  18493, 109385, 105419, 107849, 103215,   3837,
         100405,  99518,  99271, 100095,   3837, 106284, 114302, 114658,   9370,
         116679,  34187,   1773, 101469,  99417,  99741, 101910,  99436,  77540,
          14224, 105171,   3837, 100629, 106647,   9370, 102343,  98650,   3837,
          91572,  99518,  99278,  27442, 102382, 102382,   9370, 108094,  99208,
           3837, 100231, 108094,  99208,   9370, 109455,   1773, 151645, 151643]])
卫衣在春秋季节格外受欢迎，简单又百搭，可以说是时髦单品的标配了。这款卫衣采用假两件的设计，具有很强的创意感，同时又带点酷酷的街头风，适合街头风的妹子。
################################################################################################################################
167.13920497894287
请输入:
类型#裤*材质#牛仔布*风格#潮*裤型#哈伦裤*裤腰型#松紧腰*裤口#小脚
请稍等...
################################################################################################################################
{'instruction': '类型#裤*材质#牛仔布*风格#潮*裤型#哈伦裤*裤腰型#松紧腰*裤口#小脚', 'input': '', 'output': ''}
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2, 102693,      9,
         106149,      2, 100664, 102437,  51827,      9, 104040,      2, 100227,
              9, 102693,  24300,      2,  98671, 100794, 102693,      9, 102693,
         102113,  24300,      2, 100180,  99378, 102113,      9, 102693,  39426,
              2,  30709, 100037, 151645,    198, 151644]])
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裤*材质#牛仔布*风格#潮*裤型#哈伦裤*裤腰型#松紧腰*裤口#小脚<|im_end|>
<|im_start|>assistant
这款牛仔裤采用的是哈伦裤的版型，具有🚹的潮流气息，时尚的哈伦裤版型，能够很好的修饰腿部的曲线，使得腿部更加的修长。小脚的裤脚设计，能够很好的拉长腿部的曲线，同时又具有很好的保暖性，能够很好的抵御寒风的侵袭。松紧的腰带设计，更加的方便Touch，穿脱更加的方便。<|im_end|><|endoftext|>
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裤*材质#牛仔布*风格#潮*裤型#哈伦裤*裤腰型#松紧腰*裤口#小脚<|im_end|>
<|im_start|>assistant
这款牛仔裤采用的是哈伦裤的版型，具有🚹的潮流气息，时尚的哈伦裤版型，能够很好的修饰腿部的曲线，使得腿部更加的修长。小脚的裤脚设计，能够很好的拉长腿部的曲线，同时又具有很好的保暖性，能够很好的抵御寒风的侵袭。松紧的腰带设计，更加的方便Touch，穿脱更加的方便。<|im_end|><|endoftext|>
###mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid######mid###
################################################################################################################################
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裤*材质#牛仔布*风格#潮*裤型#哈伦裤*裤腰型#松紧腰*裤口#小脚<|im_end|>
<|im_start|>assistant

[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 31905, 2, 102693, 9, 106149, 2, 100664, 102437, 51827, 9, 104040, 2, 100227, 9, 102693, 24300, 2, 98671, 100794, 102693, 9, 102693, 102113, 24300, 2, 100180, 99378, 102113, 9, 102693, 39426, 2, 30709, 100037, 151645, 198, 151644, 77091, 198]
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2, 102693,      9,
         106149,      2, 100664, 102437,  51827,      9, 104040,      2, 100227,
              9, 102693,  24300,      2,  98671, 100794, 102693,      9, 102693,
         102113,  24300,      2, 100180,  99378, 102113,      9, 102693,  39426,
              2,  30709, 100037, 151645,    198, 151644,  77091,    198, 100664,
         102437, 102693, 107359,  98671, 100794,  14618,     27,  31883,   9231,
          40301,  24300, 105171,   3837,  32664, 117656,  18830, 109957, 114999,
         100154,   1773, 104392, 117656,   9370, 101538, 146937, 101538,  99338,
          68536,  23081,  45861,   3837,  91572,  99518,  16530,  99580, 121922,
         101561,   1773, 102113,  39426,  44290, 107359, 100180,  99378,   9370,
         100349, 100526,  62963,  39426,  70500,   3837, 105611, 102476,  99518,
          99580, 102647,   1773,  99739,  99580, 104070,   9370, 106294,  98650,
           1773, 151645, 151643]])
牛仔裤采用了哈伦(.<UNK>)版型的设计，对腿部有较好的修饰作用。显得腿部的纤ὖ纤细而修长，同时又不显臃肿。腰口处采用了松紧的罗纹束口设计，穿着舒适又显个性。尽显时尚的潮流感。
################################################################################################################################
103.0037031173706
请输入:
"""