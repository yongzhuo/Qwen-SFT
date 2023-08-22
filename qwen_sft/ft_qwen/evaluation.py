# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 验证评估


import logging as logger
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

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch
import jieba

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
def text_generate(request_data):
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
        response = "<|endoftext|>"
    return response
class Item(BaseModel):
    instruction: str = ""
    text: str = "1+1="
    penalty_alpha: float = 1.0
    max_new_tokens: int = 512
    temperature: float = 0.8  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.8  # 0.75
    top_k: int = 50


if __name__ == '__main__':

    text = "1+1="
    item_config = Item()
    item_config.text = text
    response = text_generate(item_config)
    print(response)

    smooth = SmoothingFunction().method1
    rouge = Rouge()
    best_bleu = 0.

    fw = open(DATA_PATH + ".qwen_eval_rouge_blue.512.json", "a+", encoding="utf-8")
    rouge_1_p, rouge_2_p, rouge_l_p = 0, 0, 0
    rouge_1_r, rouge_2_r, rouge_l_r = 0, 0, 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    total = 0
    time_start = time.time()
    datas = load_json(DATA_PATH)
    # datas = datas[:1024]
    datas = datas[:8]
    for d_json in tqdm(datas, desc="data"):
        try:
            instruction = d_json.get("instruction", "")
            text_input = d_json.get("input", "")
            text_output = d_json.get("output", "")
            # qtext, qans
            total += 1
            item_config = Item()
            item_config.instruction = instruction
            item_config.text = text_input
            text_output = " ".join(jieba.lcut(text_output))
            text_gen_str = text_generate(item_config)
            text_pred = " ".join(jieba.lcut(text_gen_str.replace("<|endoftext|>", "").replace("<|im_end|>", "")
                                            .replace("<|im_start|>", "").strip().lower()))
            line = {"input": text_input, "output": text_output.replace(" ", ""),
                    "pred": text_pred.replace(" ", "")}
            line_str = json.dumps(line, ensure_ascii=False) + "\n"
            fw.write(line_str)
            if text_pred.strip():
                scores = rouge.get_scores(hyps=text_pred, refs=text_output)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']

                rouge_1_p += scores[0]['rouge-1']['p']
                rouge_2_p += scores[0]['rouge-2']['p']
                rouge_l_p += scores[0]['rouge-l']['p']
                rouge_1_r += scores[0]['rouge-1']['r']
                rouge_2_r += scores[0]['rouge-2']['r']
                rouge_l_r += scores[0]['rouge-l']['r']

                bleu += sentence_bleu(references=[list(text_output.replace(" ", ""))],
                                      hypothesis=list(text_pred.replace(" ", "")),
                                      smoothing_function=smooth)
                mertics_i = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l,
                             'bleu': bleu,
                             'rouge-1_p': rouge_1_p, 'rouge-2_p': rouge_2_p, 'rouge-l_p': rouge_l_p,
                             'rouge-1_r': rouge_1_r, 'rouge-2_r': rouge_2_r, 'rouge-l_r': rouge_l_r, }
                if total < 5:
                    print(text_output.replace(" ", ""))
                    print(text_pred.replace(" ", ""))
                    print(mertics_i)
        except Exception as e:
            print(traceback.print_exc())
            continue
    time_end = time.time()
    lost_time = time_end - time_start
    lost_time_avg = lost_time / total
    rouge_1, rouge_2, rouge_l, bleu = rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total
    rouge_1_p, rouge_2_p, rouge_l_p = rouge_1_p / total, rouge_2_p / total, rouge_l_p / total
    rouge_1_r, rouge_2_r, rouge_l_r = rouge_1_r / total, rouge_2_r / total, rouge_l_r / total

    mertics = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu,
               "lost_time": lost_time, "lost_time_avg": lost_time_avg,
               'rouge-1_p': rouge_1_p, 'rouge-2_p': rouge_2_p, 'rouge-l_p': rouge_l_p,
               'rouge-1_r': rouge_1_r, 'rouge-2_r': rouge_2_r, 'rouge-l_r': rouge_l_r,
               }
    mertics = {k: round(v, 4) for k, v in mertics.items()}
    print(mertics, lost_time, lost_time_avg)
    fw.write(json.dumps(mertics, ensure_ascii=False) + "\n")
    fw.close()


"""
# nohup python evaluation.py > tc.evaluation.py.log 2>&1 &
# tail -n 1000  -f tc.evaluation.py.log
# |myz|
"""


### log 测试10个sample
"""
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裙*裙下摆#弧形*裙腰型#高腰*裙长#半身裙*裙款式#不规则*裙款式#收腰<|im_end|>
<|im_start|>assistant
这款半身裙采用了高腰的设计，能够提高腰线，打造大长腿。不规则的裙摆，显得个性十足Honda。腰部采用了收腰的设计，能够凸显腰身，更显女性的柔美。下摆采用了弧形的设计，能够修饰身材。<|im_end|><|endoftext|>
data:  62%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                               | 5/8 [02:42<01:37{'instruction': '', 'text': '类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳', 'penalty_alpha': 1.0, 'max_new_tokens': 512, 'temperature': 0.8, 'do_sample': True, 'num_top_p': 0.8, 'top_k': 50}
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2,  17447,  99741,
              9,  40301,  24300,      2, 109285,      9,  40301,  24300,      2,
          99580, 102372,      9, 108108,      2, 108236,      9,  99741, 112453,
              2, 113727,      9,  99741, 102885,  24300,      2, 117801, 102885,
              9,  99741, 108756,      2,  99950, 106491, 151645,    198, 151644,
          77091,    198]])
tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
        151645,    198, 151644,    872,    198,  31905,      2,  17447,  99741,
             9,  40301,  24300,      2, 109285,      9,  40301,  24300,      2,
         99580, 102372,      9, 108108,      2, 108236,      9,  99741, 112453,
             2, 113727,      9,  99741, 102885,  24300,      2, 117801, 102885,
             9,  99741, 108756,      2,  99950, 106491, 151645,    198, 151644,
         77091,    198, 113727, 107359, 109285,   9370,  40301,  24300,  70500,
          3837, 100629, 106515,  99894,  99580, 102372, 105005,   3837,  91572,
        104468, 114999, 105568, 108236,   1773, 117801, 102885,   9370, 108756,
         70500,   3837,  73670, 106515, 100890, 108808,   9370, 116799,  99894,
          3837, 108251, 101538,   9567,   9370,  99493, 102768, 108236,   3837,
        102885,  39426, 107359,  99950, 106491,   9370, 145174,  99213,  39426,
          3837,  73670, 101972, 105053, 100180,  99378,  26381,   3837, 101147,
        102556,   1773, 151645, 151643])
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳<|im_end|>
<|im_start|>assistant
衬衫采用了宽松的版型设计，具有遮肉显瘦的效果，同时还可以修饰身材线条。泡泡袖的款式设计，可以遮挡手臂的赘肉，凸显纤 gather的双臂线条，袖口采用了抽绳的쉘领口，可以自由调节松紧度，方便实用。<|im_end|><
data:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                          | 6/8 [03:17<01:06{'instruction': '', 'text': '类型#裙*材质#蕾丝*风格#宫廷*图案#刺绣*图案#蕾丝*裙型#大裙摆*裙下摆#花边*裙袖型#泡泡袖', 'penalty_alpha': 1.0, 'max_new_tokens': 512, 'temperature': 0.8, 'do_sample': Trus': 1, 'top_p': 0.8, 'top_k': 50}
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2, 102807,      9,
         106149,      2, 106291,  99691,      9, 104040,      2, 117354,      9,
         108108,      2, 100253, 103084,      9, 108108,      2, 106291,  99691,
              9, 102807,  24300,      2,  26288, 102807, 100805,      9, 102807,
          16872, 100805,      2,  99232,  99319,      9, 102807, 102885,  24300,
              2, 117801, 102885, 151645,    198, 151644,  77091,    198]])
tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
        151645,    198, 151644,    872,    198,  31905,      2, 102807,      9,
        106149,      2, 106291,  99691,      9, 104040,      2, 117354,      9,
        108108,      2, 100253, 103084,      9, 108108,      2, 106291,  99691,
             9, 102807,  24300,      2,  26288, 102807, 100805,      9, 102807,
         16872, 100805,      2,  99232,  99319,      9, 102807, 102885,  24300,
             2, 117801, 102885, 151645,    198, 151644,  77091,    198, 117354,
         99208,  26288, 100805, 102807,   3837, 102807, 100805, 101235,   3837,
         99999,  99621,  99793,  99165, 104712, 104815,   3837,  17447,  99369,
         95256,  20412, 118066,   9370,   3837,  18830, 106291,  99691,  99232,
         99319,   3837,  99491, 106662,   1773, 117801, 102885, 105171,   3837,
        108251, 105336,  98650,   3837, 117024,   9370, 100253, 103084,  70500,
          3837, 104392, 106110,   3837, 105098,  13072, 109970,  98650,   1773,
        151645, 151643])
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裙*材质#蕾丝*风格#宫廷*图案#刺绣*图案#蕾丝*裙型#大裙摆*裙下摆#花边*裙袖型#泡泡袖<|im_end|>
<|im_start|>assistant
宫廷风大摆裙，裙摆很大，所以穿起来很飘逸，上半身是透视的，有蕾丝花边，非常优雅。泡泡袖的设计，凸显少女感，胸前的刺绣设计，显得精致，很有名媛感。<|im_end|><|endoftext|>
data:  88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                     | 7/8 [03:50<00:33{'instruction': '', 'text': '类型#裤*版型#显瘦*颜色#黑色*风格#简约*裤长#九分裤', 'penalty_alpha': 1.0, 'max_new_tokens': 512, 'temperature': 0.8, 'do_sample': True, 'num_beams': 1, 'top_p': 0.8, 'to
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198,  31905,      2, 102693,      9,
          40301,  24300,      2,  99580, 102372,      9, 102284,      2, 104723,
              9, 104040,      2, 109962,      9, 102693,  45861,      2,  99609,
          17177, 102693, 151645,    198, 151644,  77091,    198]])
tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
        151645,    198, 151644,    872,    198,  31905,      2, 102693,      9,
         40301,  24300,      2,  99580, 102372,      9, 102284,      2, 104723,
             9, 104040,      2, 109962,      9, 102693,  45861,      2,  99609,
         17177, 102693, 151645,    198, 151644,  77091,    198, 104070,  99271,
        100095,   9370, 104723,  38176,   3837, 109962, 105797,   3837, 114660,
        104402,   1773, 117538,   9370,  40301,  24300,  70500,   3837,  99934,
         39762, 101099, 109395,   3837, 103186,  99580, 100041,  73145,  23081,
         45861,   9370, 100447,  82699,   1773,  99609,  17177, 102693,   9370,
         98402,   3837, 105722, 101538,  99338,   9370, 100037, 121601,   3837,
         72225,  45861, 116770,   3837,  33126,  99580,  44636, 100428,   1773,
        151645, 151643])
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
类型#裤*版型#显瘦*颜色#黑色*风格#简约*裤长#九分裤<|im_end|>
<|im_start|>assistant
时尚百搭的黑色系，简约大气，易于搭配。修身的版型设计，贴合身体曲线，衬显笔直修长的腿形。九分裤的长度，露出纤细的脚踝，拉长身形，更显高挑。<|im_end|><|endoftext|>
data: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [04:21<00:00
{'rouge-1': 0.3485, 'rouge-2': 0.0792, 'rouge-l': 0.1937, 'bleu': 0.0832, 'lost_time': 262.0019, 'lost_time_avg': 32.7502, 'rouge-1_p': 0.4064, 'rouge-2_p': 0.0955, 'rouge-l_p': 0.2291, 'rouge-1_r':uge-2_r': 0.0695, 'rouge-l_r': 0.1719} 262.0019030570984 32.7502378821373
"""


