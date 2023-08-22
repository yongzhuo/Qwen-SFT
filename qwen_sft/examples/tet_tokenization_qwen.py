# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenization classes for QWen."""

import base64
import logging
import os
import unicodedata
from typing import Collection, Dict, List, Set, Tuple, Union

import tiktoken
from transformers import PreTrainedTokenizer, AddedToken

logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "qwen.tiktoken"}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (
    ENDOFTEXT,
    IMSTART,
    IMEND,
) + EXTRAS


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

class QWenTokenizer(PreTrainedTokenizer):
    """QWen tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors="replace",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.errors = errors  # how to handle errors in decoding

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        self.special_tokens = {
            token: index
            for index, token in enumerate(
                SPECIAL_TOKENS, start=len(self.mergeable_ranks)
            )
        }

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return self.mergeable_ranks

    def convert_tokens_to_ids(
        self, tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        if not special_tokens and new_tokens:
            raise ValueError('Adding regular tokens is not supported')
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS:
                raise ValueError('Adding unknown special tokens is not supported')
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "qwen.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


if __name__ == '__main__':
    ee = 0
    ENDOFTEXT = "<|endoftext|>"
    IMSTART = "<|im_start|>"
    IMEND = "<|im_end|>"
    SPECIAL_TOKENS = [ENDOFTEXT, IMSTART, IMEND]
    path_vocab = "qwen.tiktoken"
    LLM_tokenizer = QWenTokenizer.from_pretrained(path_vocab)
    # text = "<|im_start|>user\n{问题}<|im_end|>\n<|im_start|>"
    # text_5 = LLM_tokenizer.encode(text)
    # print(text_5)
    # text = "1+1=?"
    # text_4 = LLM_tokenizer.encode(text)
    # print(text_4)
    # text = "<|im_start|>1+1=?<|im_end|><|endoftext|>"
    # text_3 = LLM_tokenizer.encode(text)
    # print(text_3)
    # text_2 = LLM_tokenizer.tokenize(text)
    # print(text_2)
    for t in SPECIAL_TOKENS:
        text_1 = LLM_tokenizer.convert_tokens_to_ids(t)
        print(text_1)
    # text_chat = make_context(tokenizer=LLM_tokenizer, query="你好", system="你是一个有用的机器人")
    # print(text_chat)
    """
    [151644, 872, 198, 90, 86119, 92, 151643, 198, 151644]
    [16, 10, 16, 19884]
    [151644, 16, 10, 16, 19884, 151643, 151645]
    ['<|im_start|>', b'1', b'+', b'1', b'=?', '<|im_end|>', '<|endoftext|>']
    151643
    151644
    151645
    ('<|im_start|>system\n你是一个有用的机器人<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n', 
    [151644, 8948, 198, 56568, 101909, 115405, 104354, 151643, 198, 151644, 872, 198, 108386, 151643, 198, 151644, 77091, 198])
    """

    text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
解释为什么下面的分数等于 1/4    解释为什么下面的分数等于 1/4，4/16<|im_end|>
<|im_start|>assistant
"""
    text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
解释为什么下面的分数等于 1/4    解释为什么下面的分数等于 1/4，4/16<|im_end|>
<|im_start|>assistant
"""
    print(LLM_tokenizer.encode(text))
    print(LLM_tokenizer.tokenize(text))

    res_1 = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
             151645, 198, 151644, 872, 198, 104136, 100678, 100431, 9370,
             103190, 107106, 220, 16, 14, 19,
             262, 85122, 68862,
             100678, 100431, 9370, 103190, 107106, 220, 16, 14, 19, 3837,
            19, 14, 16, 21, 151645, 198, 151644, 77091, 198]

    res_1 = [151644, 8948, 198, 2610, 525, 264, 10950, 17847,
             151645, 198, 151644, 872, 198, 104136, 100678, 100431, 9370,
             103190, 107106, 220, 16, 14, 19,
             262, 85122, 68862,
             100678, 100431, 9370, 103190, 107106, 220, 16, 14, 19, 3837,
             19, 14, 16, 21, 151645, 198, 151644, 77091, 198]
    res_1 = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
             151645, 198, 151644, 872, 198, 104136, 100678, 100431, 9370,
             103190, 107106, 220, 16, 14, 19,
             262, 85122, 68862,
             100678, 100431, 9370, 103190, 107106, 220, 16, 14, 19, 3837, 19, 14, 16, 21, 151645, 198, 151644, 77091, 198]

    res_2 = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
             151645, 198, 151644, 872, 198, 104136, 100678, 100431, 9370,
             103190, 107106, 220, 16, 14, 19,
             197, 104136,
             100678, 100431, 9370, 103190, 107106, 220, 16, 14, 19, 3837,
             19, 14, 16, 21, 151645, 198, 151644, 77091, 198]
    count = 0
    print("#"*128)
    for x, y in zip(res_1, res_2):
        count += 1
        if x != y:
            print(x, y, x==y)
            print(count)
    ids = [262, 85122, 68862, 104136, 13, 197, 104136,]
    res = LLM_tokenizer.convert_ids_to_tokens(ids)
    print(res)
    from qwen_sft.models.qwen.qwen_generation_utils import make_context
    system = "You are a helpful assistant."
    query = "解释为什么下面的分数等于 1/4    解释为什么下面的分数等于 1/4，4/16"
    raw_text, context_tokens = make_context(
                LLM_tokenizer,
                query.strip(),
                system=system,
                max_window_size=6144
    )
    print(raw_text)
    print(context_tokens)

    MAX_LENGTH_Q = 256
    MAX_LENGTH_A = 256
    ID_PAD = 151643
    ID_EOS = 151643  # endoftext
    ID_SOP = 151644  # start
    ID_EOP = 151645  # end
    def generate_prompt(data_point, is_logger=False):
        system_str = "You are a helpful assistant."
        prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(system_str)
        text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
        text_out = data_point.get("output", "")
        prompt_text_1 = prompt_system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        prompt_text_2 = "{}<|im_end|><|endoftext|>"
        text_1 = prompt_text_1.format(text_input.strip())
        text_2 = prompt_text_2.format(text_out)
        # end with <|im_end|><|endoftext|>
        x = LLM_tokenizer.encode(text_1)
        y = LLM_tokenizer.encode(text_2)
        if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
            x = x[:MAX_LENGTH_Q]
            y = y[:MAX_LENGTH_A]
        if not x:
            x = [ID_SOP, ID_PAD, ID_EOP, ID_SOP]
        if x[-2] != ID_EOP:  # 截断补上
            x += [ID_EOP]
        if x[-1] != ID_SOP:
            x += [ID_SOP]
        if not y:
            y = [ID_PAD, ID_EOP, ID_EOS]
        if y[-2] != ID_EOP:
            y += [ID_EOP]
        if y[-1] != ID_EOS:
            y += [ID_EOS]
        out = {"input_ids": x, "labels": y}
        if is_logger:
            print(text_1)
            print(text_2)
            print(out)
        return out
    def generate_prompt_new_1(data_point, is_logger=False):
        from qwen_sft.models.qwen.qwen_generation_utils import make_context
        system = "You are a helpful assistant."
        query = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
        raw_text, context_tokens = make_context(
            LLM_tokenizer,
            query.strip(),
            history=[],
            system=system,
            max_window_size=6144
        )
        out = {"input_ids": context_tokens}
        return out
    ques_dict = {"instruction": "1+1=", "input": "", "output": ""}
    res_1 = generate_prompt_new_1(ques_dict)
    res_2 = generate_prompt(ques_dict)
    print(res_1)
    print(res_2)

    print(LLM_tokenizer.encode("\n"))
