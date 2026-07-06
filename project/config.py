"""
全局配置文件 —— 所有路径、超参、常量集中管理
"""
import os
import torch

# ======================== 路径 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.join(BASE_DIR, "cv")
FUSION_DIR = os.path.join(BASE_DIR, "fusion")
NLP_DIR = os.path.join(BASE_DIR, "nlp")
DATA_DIR = BASE_DIR

# 数据集
DATASET_CSV = os.path.join(FUSION_DIR, "dataset.csv")

# CV 权重
CV_WEIGHT_PATH = os.path.join(CV_DIR, "rafdb_emotion_model.pth")

# ======================== CV 参数 ========================
CV_IMAGE_SIZE = 224
CV_FEATURE_DIM = 512        # ResNet18 倒数第二层输出维度
CV_NUM_CLASSES = 7           # RAF-DB 的 7 种表情
CV_EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CV_IMAGENET_MEAN = [0.485, 0.456, 0.406]
CV_IMAGENET_STD = [0.229, 0.224, 0.225]

# ======================== NLP 参数 ========================
NLP_FEATURE_DIM = 768         # RoBERTa [CLS] 语义向量
NLP_CLASS_NAMES = ["negative", "neutral", "positive"]  # 负面 / 中立 / 正面
NLP_MODEL_PATH = os.path.join(NLP_DIR, "sentiment_model")
NLP_MAX_LENGTH = 128

# ======================== 融合模型参数 ========================
FUSION_NUM_CLASSES = 5                                    # 喜 / 怒 / 哀 / 惊 / 讽
FUSION_CLASS_NAMES = ["happy", "angry", "sad", "surprise", "sarcasm"]
FUSION_HIDDEN_DIM = 256
FUSION_GATE_HIDDEN = 128

# 冲突感知专用参数
CONFLICT_AWARE = True         # 是否启用冲突感知模块
EXPERT_HIDDEN_DIM = 64        # 单模态专家分类器的隐藏层大小

# ======================== 训练参数 ========================
BATCH_SIZE = 2               # 当前数据量小，设小一点
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# ======================== 设备 ========================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
