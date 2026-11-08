"""장치 선택, 실제 모델 로드, 모의(mock) 모델 생성."""

import json

import torch

from .config import MODEL_ID, MODEL_REVISION, RunConfig


def pick_device(name: str = "auto") -> tuple[torch.device, torch.dtype]:
    """장치와 정밀도를 고른다. CUDA·MPS는 bfloat16, CPU는 float32."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    dtype = torch.float32 if name == "cpu" else torch.bfloat16
    return torch.device(name), dtype


def load_real(device: torch.device, dtype: torch.dtype):
    """Hugging Face에서 Qwen2.5-0.5B-Instruct를 내려받아(처음 한 번) 올린다."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    return tok, model.to(device)


def build_mock(cfg: RunConfig):
    """네트워크 없이 쓰는 '진짜와 같은 구조의 장난감 모델'.

    - 토크나이저: 학습 데이터 문장으로 작은 BPE를 즉석에서 학습 (ChatML 템플릿은 Qwen과 같은 모양)
    - 모델: 무작위 초기화한 Qwen2 구조(층 2개). 실제 모델과 같은 클래스라 LoRA 대상 층 이름이 같다.
    답변 품질은 의미 없고, '전체 파이프라인이 도는지'만 확인하는 용도다.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    rows = [json.loads(line) for line in cfg.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [m["content"] for r in rows for m in r["messages"]]
    texts += json.loads(cfg.eval_prompts.read_text(encoding="utf-8"))

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    bpe = Tokenizer(models.BPE(unk_token=None))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=800,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    bpe.train_from_iterator(texts + ["user assistant system"], trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    torch.manual_seed(cfg.seed)  # 같은 seed → 같은 무작위 초기 가중치
    conf = Qwen2Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    return tok, Qwen2ForCausalLM(conf)
