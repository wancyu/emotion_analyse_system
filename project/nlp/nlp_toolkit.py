"""
NLP 文本情感分析工具箱 —— chinese-roberta-wwm-ext
从 hdp/last/nlp_toolkit.py 迁移，对齐 final/cv/vision_toolkit.py 设计风格。

接口:
  - 模块级常量 (SENTIMENT_LABELS)
  - 加载函数 (load_sentiment_model, load_feature_extractor, load_tokenizer)
  - 推理函数 (predict_sentiment, extract_sentiment)
  - 特征提取 (extract_text_features)
"""
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==================== 常量 ====================
SENTIMENT_LABELS = ["negative", "neutral", "positive"]  # 负面 / 中立 / 正面
MAX_LENGTH = 128
FEATURE_DIM = 768  # [CLS] hidden size


# ==================== 设备 ====================
def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==================== 模型构建 & 加载 ====================
def load_sentiment_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """加载完整的情感分类模型（含 3 分类头），用于端到端预测"""
    device = device or get_device()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model.to(device).eval()


def load_tokenizer(model_path: str):
    """加载分词器"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer not found: {model_path}")
    return AutoTokenizer.from_pretrained(model_path)


def load_feature_extractor(
    model_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """加载特征提取器 —— 保留 base model，输出 [CLS] 768d 语义向量"""
    device = device or get_device()
    model = load_sentiment_model(model_path, device)
    # 返回 base model (RoBERTa)，去掉分类头
    return model.to(device).eval()


# ==================== 预处理 ====================
def preprocess_text(text: str, tokenizer, device: torch.device) -> dict:
    """文本 → tokenized inputs"""
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )


# ==================== 推理 ====================
@torch.inference_mode()
def extract_sentiment(
    text: str,
    model: nn.Module,
    tokenizer,
    device: Optional[torch.device] = None,
) -> list:
    """返回 3 分类概率 [neg, neu, pos]"""
    device = device or next(model.parameters()).device
    inputs = preprocess_text(text, tokenizer, device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    return probs.cpu().tolist()


@torch.inference_mode()
def predict_sentiment(
    text: str,
    model: nn.Module,
    tokenizer,
    device: Optional[torch.device] = None,
) -> dict:
    """端到端情感预测 → {label_name, label_id, confidence, probabilities}"""
    probabilities = extract_sentiment(text, model, tokenizer, device)
    label_id = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    return {
        "label_id": label_id,
        "label_name": SENTIMENT_LABELS[label_id],
        "confidence": probabilities[label_id],
        "probabilities": dict(zip(SENTIMENT_LABELS, probabilities)),
    }


# ==================== 特征提取 ====================
@torch.inference_mode()
def extract_text_features(
    text: str,
    model: nn.Module,
    tokenizer,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """提取 [CLS] token 的 768 维语义向量，喂给融合模型"""
    device = device or next(model.parameters()).device
    inputs = preprocess_text(text, tokenizer, device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 绕过分类头，直接取 base model 输出
    outputs = model.bert(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_embedding.cpu()


# ==================== 封装类（对接融合训练） ====================
class NLPFeatureExtractor:
    """封装 NLP 模型 + 分词器，提供统一的 __call__ 接口，供 EmotionDataset 调用"""

    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, text: str) -> torch.Tensor:
        """text → [768] 语义向量"""
        feat = extract_text_features(text, self.model, self.tokenizer, self.device)
        return feat.squeeze(0)  # [768]
