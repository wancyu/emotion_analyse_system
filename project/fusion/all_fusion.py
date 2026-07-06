"""
多模态融合模型三件套 —— CV(512d) + NLP(768d) → 情感分类(4类)

┌─────────────────────────────────────────────────────────────┐
│ 1. EarlyFusion  (早期融合)                                   │
│    CV + NLP 直接拼接 → 单分类器                              │
│    特点: 最简单，参数少，让模型自己学交互                      │
│                                                             │
│ 2. LateFusion   (晚期融合)                                   │
│    CV/NLP 各自独立预测 → 加权投票                             │
│    特点: 单模态可独立工作，可解释性强                         │
│                                                             │
│ 3. GatedFusion  (门控融合) ← 当前方案                        │
│    学一个 α 动态调节 CV/NLP 权重 → 加权拼接 → 分类器         │
│    特点: 自适应，CV 不可靠时自动信任 NLP，反之亦然            │
└─────────────────────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  通用参数
# ============================================================
CV_DIM = 512
NLP_DIM = 768
NUM_CLASSES = 5        # 喜/怒/哀/惊/讽
HIDDEN_DIM = 256
CLASS_NAMES = ["happy", "angry", "sad", "surprise", "sarcasm"]


# ============================================================
#  1. EarlyFusion —— 早期融合
# ============================================================
class EarlyFusion(nn.Module):
    """
    思想: CV 和 NLP 特征在最早期就混在一起，交给一个网络处理。

    流程:
      CV(512d) ──┐
                  ├── concat(1280d) ──→ MLP ──→ logits(4)
      NLP(768d) ─┘

    优点: 简单粗暴，参数少，分类器能看到全部原始信息
    缺点: CV 出错时 NLP 也无法挽救，因为特征一开始就混了
    """

    def __init__(self, cv_dim=CV_DIM, nlp_dim=NLP_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.cv_dim = cv_dim
        self.nlp_dim = nlp_dim

        self.classifier = nn.Sequential(
            nn.Linear(cv_dim + nlp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, cv_feat, nlp_feat):
        """
        Args:
            cv_feat:  [B, 512]
            nlp_feat: [B, 768]
        Returns:
            logits: [B, num_classes]
        """
        concat = torch.cat((cv_feat, nlp_feat), dim=1)  # [B, 1280]
        return self.classifier(concat)

    @torch.inference_mode()
    def predict(self, cv_feat, nlp_feat, class_names=None):
        """单样本推理"""
        if cv_feat.dim() == 1:
            cv_feat = cv_feat.unsqueeze(0)
        if nlp_feat.dim() == 1:
            nlp_feat = nlp_feat.unsqueeze(0)

        logits = self.forward(cv_feat, nlp_feat)
        probs = F.softmax(logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()

        return {
            "label_id": pred_id,
            "confidence": probs[pred_id].item(),
            "probabilities": probs.tolist(),
        }


# ============================================================
#  2. LateFusion —— 晚期融合
# ============================================================
class LateFusion(nn.Module):
    """
    思想: CV 和 NLP 各自独立判断，最后投票。就像两个专家各看各的，再综合意见。

    流程:
      CV(512d)  ──→ cv_expert ──→ cv_logits(4)  ──┐
                                                    ├── 加权平均 ──→ final_logits(4)
      NLP(768d) ──→ nlp_expert ──→ nlp_logits(4) ──┘

    优点: 可解释性强（能看到每个模态单独的判断），单模态可独立部署
    缺点: 模态间的交互只在最后一步，没有深层融合
    """

    def __init__(self, cv_dim=CV_DIM, nlp_dim=NLP_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.num_classes = num_classes

        # CV 独立专家
        self.cv_expert = nn.Sequential(
            nn.Linear(cv_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # NLP 独立专家
        self.nlp_expert = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # 可学习的模态权重（初始各 0.5）
        self.cv_weight = nn.Parameter(torch.tensor(0.5))
        self.nlp_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, cv_feat, nlp_feat):
        """
        Returns:
            final_logits: [B, num_classes]
            cv_logits:    [B, num_classes]
            nlp_logits:   [B, num_classes]
            weights:      (w_cv, w_nlp) 归一化后的权重
        """
        cv_logits = self.cv_expert(cv_feat)        # [B, C]
        nlp_logits = self.nlp_expert(nlp_feat)     # [B, C]

        # 归一化权重 (softmax 保证 sum=1)
        w = torch.softmax(torch.stack([self.cv_weight, self.nlp_weight]), dim=0)
        w_cv, w_nlp = w[0], w[1]

        final_logits = w_cv * cv_logits + w_nlp * nlp_logits

        return final_logits, cv_logits, nlp_logits, (w_cv.item(), w_nlp.item())

    @torch.inference_mode()
    def predict(self, cv_feat, nlp_feat, class_names=None):
        """单样本推理，返回 CV/NLP 各自判断 + 最终结果"""
        if cv_feat.dim() == 1:
            cv_feat = cv_feat.unsqueeze(0)
        if nlp_feat.dim() == 1:
            nlp_feat = nlp_feat.unsqueeze(0)

        final, cv_logits, nlp_logits, weights = self.forward(cv_feat, nlp_feat)

        final_probs = F.softmax(final, dim=1)[0]
        cv_probs = F.softmax(cv_logits, dim=1)[0]
        nlp_probs = F.softmax(nlp_logits, dim=1)[0]

        return {
            "final": {
                "label_id": torch.argmax(final_probs).item(),
                "confidence": final_probs.max().item(),
                "probabilities": final_probs.tolist(),
            },
            "cv_only": {
                "label_id": torch.argmax(cv_probs).item(),
                "confidence": cv_probs.max().item(),
                "probabilities": cv_probs.tolist(),
            },
            "nlp_only": {
                "label_id": torch.argmax(nlp_probs).item(),
                "confidence": nlp_probs.max().item(),
                "probabilities": nlp_probs.tolist(),
            },
            "cv_weight": weights[0],
            "nlp_weight": weights[1],
        }


# ============================================================
#  3. GatedFusion —— 门控融合（当前方案）
# ============================================================
class GatedFusion(nn.Module):
    """
    思想: 根据 CV+NLP 的拼接信息，动态决定该信谁多少。不是固定的权重，
         而是每个样本自己算出来的 α。

    流程:
      CV(512d) ──┐
                  ├── concat ──→ gate ──→ α (0~1)
      NLP(768d) ─┘                    ──→ weighted_concat ──→ classifier

      weighted_nlp = nlp × α          (α 大 → 更信文本)
      weighted_cv  = cv × (1-α)       (α 小 → 更信图片)

    优点: 自适应，对单模态噪声鲁棒
    缺点: 需要足够数据学出有意义的 α
    """

    def __init__(self, cv_dim=CV_DIM, nlp_dim=NLP_DIM, num_classes=NUM_CLASSES,
                 gate_hidden=128, classifier_hidden=HIDDEN_DIM):
        super().__init__()
        concat_dim = cv_dim + nlp_dim

        # 门控网络: 看拼接特征 → 决定 α
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, cv_feat, nlp_feat):
        """
        Returns:
            logits: [B, num_classes]
            alpha:  [B, 1]  门控系数 (每个样本不同)
        """
        raw_concat = torch.cat((cv_feat, nlp_feat), dim=1)
        alpha = self.gate(raw_concat)                    # [B, 1]

        weighted_nlp = nlp_feat * alpha
        weighted_cv = cv_feat * (1.0 - alpha)
        fused = torch.cat((weighted_cv, weighted_nlp), dim=1)

        logits = self.classifier(fused)
        return logits, alpha

    @torch.inference_mode()
    def predict(self, cv_feat, nlp_feat, class_names=None):
        """单样本推理"""
        if cv_feat.dim() == 1:
            cv_feat = cv_feat.unsqueeze(0)
        if nlp_feat.dim() == 1:
            nlp_feat = nlp_feat.unsqueeze(0)

        logits, alpha = self.forward(cv_feat, nlp_feat)
        probs = F.softmax(logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()

        return {
            "label_id": pred_id,
            "confidence": probs[pred_id].item(),
            "probabilities": probs.tolist(),
            "alpha": alpha[0].item(),
            "trust_nlp": alpha[0].item() > 0.5,
        }


# ============================================================
#  模型注册表
# ============================================================
MODELS = {
    "early": EarlyFusion,
    "late": LateFusion,
    "gated": GatedFusion,
}


def build_fusion(model_type="gated", **kwargs):
    """工厂函数: build_fusion('early'), build_fusion('late'), build_fusion('gated')"""
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODELS.keys())}")
    return MODELS[model_type](**kwargs)
