"""
NLP 文本模块 —— 基于 chinese-roberta-wwm-ext 的文本情感特征提取与识别
"""
from .nlp_toolkit import (
    SENTIMENT_LABELS,
    get_device,
    preprocess_text,
    load_sentiment_model,
    load_tokenizer,
    load_feature_extractor,
    extract_sentiment,
    predict_sentiment,
    extract_text_features,
    NLPFeatureExtractor,
)
