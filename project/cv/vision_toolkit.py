"""
CV 视觉工具箱 —— ResNet18 人脸表情识别与特征提取
从 xt/last/vision_toolkit.py 迁移，保持与原版一致。
"""
import os
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(img_path: str, device: torch.device, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """将图片转化为模型输入 tensor [1, 3, 224, 224]"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    tensor = build_transform(image_size)(image).unsqueeze(0)
    return tensor.to(device)


def preprocess_array(img_array, device: torch.device, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """numpy BGR array → tensor [1, 3, 224, 224]，用于摄像头实时推理"""
    import cv2
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    tensor = build_transform(image_size)(image).unsqueeze(0)
    return tensor.to(device)


def build_emotion_model(num_classes: int = 7) -> nn.Module:
    """构建 ResNet18 + 自定义分类头"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _torch_load_state_dict(weight_path: str, device: torch.device):
    """兼容不同 torch 版本的 state_dict 读取。"""
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def load_emotion_model(
    weight_path: str, device: Optional[torch.device] = None, num_classes: int = 7
) -> nn.Module:
    """加载完整分类模型（用于端到端预测）"""
    device = device or get_device()
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    model = build_emotion_model(num_classes)
    state_dict = _torch_load_state_dict(weight_path, device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_feature_extractor(
    weight_path: str, device: Optional[torch.device] = None, num_classes: int = 7
) -> nn.Module:
    """加载特征提取器 —— 砍掉 FC 层，输出 [1, 512]"""
    device = device or get_device()
    model = build_emotion_model(num_classes)
    state_dict = _torch_load_state_dict(weight_path, device)
    model.load_state_dict(state_dict)
    extractor = nn.Sequential(*list(model.children())[:-1])
    return extractor.to(device).eval()


@torch.inference_mode()
def predict_emotion_probabilities(
    img_path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
    image_size: int = IMAGE_SIZE,
) -> List[float]:
    device = device or next(model.parameters()).device
    tensor = preprocess_image(img_path, device, image_size=image_size)
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    return probabilities.detach().cpu().tolist()


@torch.inference_mode()
def predict_emotion(
    img_path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """端到端表情预测 → {label_name, confidence, probabilities}"""
    probabilities = predict_emotion_probabilities(
        img_path=img_path,
        model=model,
        device=device,
        image_size=image_size,
    )
    label_id = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    return {
        "label_id": label_id,
        "label_name": EMOTION_NAMES[label_id],
        "confidence": probabilities[label_id],
        "probabilities": probabilities,
    }


@torch.inference_mode()
def predict_emotion_from_array(
    img_array,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> dict:
    """numpy BGR 数组 → 表情预测（摄像头专用，零文件IO）"""
    device = device or next(model.parameters()).device
    tensor = preprocess_array(img_array, device)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
    label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "label_id": label_id,
        "label_name": EMOTION_NAMES[label_id],
        "confidence": probs[label_id],
        "probabilities": probs,
    }


@torch.inference_mode()
def extract_face_features(
    img_path: str,
    extractor_model: nn.Module,
    device: Optional[torch.device] = None,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """提取人脸 512 维特征向量 [1, 512]"""
    device = device or next(extractor_model.parameters()).device
    tensor = preprocess_image(img_path, device, image_size=image_size)
    features = extractor_model(tensor)
    return features.flatten(1)
