# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/8/4 15:11
# @author  : Mo
# @function:



import traceback
import random
import time
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
CUDA_VISIBLE_DEVICES = "-1"
USE_TORCH = "1"
CPU_NUMS = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


# PATH_PRERAIN_MODEL = "Qwen/Qwen-7B-Chat"
PATH_PRERAIN_MODEL = "/pretrain_models/pytorch/Qwen_Qwen-7B-Chat/"

tokenizer = AutoTokenizer.from_pretrained(PATH_PRERAIN_MODEL, trust_remote_code=True)
# use bf16
model = AutoModelForCausalLM.from_pretrained(PATH_PRERAIN_MODEL,
                                             trust_remote_code=True).bfloat16().eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained(PATH_PRERAIN_MODEL, device_map="auto", trust_remote_code=True, fp16=True).eval()
# use fp32
# model = AutoModelForCausalLM.from_pretrained(PATH_PRERAIN_MODEL,
#                                              device_map="auto",
#                                              trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(PATH_PRERAIN_MODEL,
                                                           trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# 第一轮对话 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 第三轮对话 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》


if __name__ == '__main__':
    import traceback
    import time

    data_dict = {"instruction": "解释为什么下面的分数等于 1/4",
                 "input": "解释为什么下面的分数等于 1/4，4/16",
                 "output": "分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。"
                 }
    response, history = model.chat(tokenizer, data_dict.get("instruction")
                                   +"\t"+data_dict.get("input"), history=history)
    print(response)
    while True:
        try:
            time_start = time.time()
            history = []
            print("请输入:")
            ques = input()
            print("请稍等...")

            if ques.strip().upper() == "CLEAR":
                history = []
                print("clear ok")
                continue
            else:
                ques_dict = {"instruction": ques, "input": "", "output": ""}
                # ques_dict = ques
                response, history = model.chat(tokenizer, ques, history=history)
                print(response)
            print(time.time() - time_start)
        except Exception as e:
            print(traceback.print_exc())
            print(str(e))
