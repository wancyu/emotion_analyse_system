"""
门控多模态融合模型 —— CV(512d) + NLP(768d) → Gated Fusion → 情感分类

包含两个版本:
  - GatedMultimodalFusion  : 基础门控（基线对照）
  - SarcasmAwareFusion     : 冲突感知版（讽刺/反差分析）
"""
import os
import sys

# 确保项目根目录在 sys.path 中
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CV_FEATURE_DIM, NLP_FEATURE_DIM,
    FUSION_GATE_HIDDEN, FUSION_HIDDEN_DIM,
    EXPERT_HIDDEN_DIM,
)


# ============================================================
#  基础版: GatedMultimodalFusion（留作基线对照）
# ============================================================
class GatedMultimodalFusion(nn.Module):
    """
    门控注意力融合（基线版）:
      - 拼接 CV+NLP 特征，通过一个小网络学出门控系数 α
      - α 控制 NLP 特征的权重，(1-α) 控制 CV 特征的权重
      - 加权后的特征再次拼接，送入分类器
    """

    def __init__(
        self,
        cv_dim: int = CV_FEATURE_DIM,
        nlp_dim: int = NLP_FEATURE_DIM,
        num_classes: int = 4,
        gate_hidden: int = FUSION_GATE_HIDDEN,
        classifier_hidden: int = FUSION_HIDDEN_DIM,
    ):
        super(GatedMultimodalFusion, self).__init__()
        concat_dim = cv_dim + nlp_dim

        self.gate = nn.Sequential(
            nn.Linear(concat_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, cv_feat, nlp_feat):
        raw_concat = torch.cat((cv_feat, nlp_feat), dim=1)
        alpha = self.gate(raw_concat)

        weighted_nlp = nlp_feat * alpha
        weighted_cv = cv_feat * (1.0 - alpha)
        fused_feat = torch.cat((weighted_cv, weighted_nlp), dim=1)

        logits = self.classifier(fused_feat)
        return logits, alpha


# ============================================================
#  升级版: SarcasmAwareFusion（冲突感知）
# ============================================================
class SarcasmAwareFusion(nn.Module):
    """
    冲突感知多模态融合模型

    核心思路:
      ┌─────────────────────────────────────────────────┐
      │  CV 512d  ──→ cv_expert    ──→ cv_logits       │
      │                         ↘                      │
      │  NLP 768d ──→ nlp_expert  ──→ nlp_logits       │
      │                         ↗    ↓                  │
      │              conflict = |cv_logits - nlp_logits| │
      │                         ↓                       │
      │              fusion_classifier(cv, nlp, conflict)│
      │                         ↓                       │
      │              final_logits + conflict_score       │
      └─────────────────────────────────────────────────┘

    输出解读:
      - cv_logits     : 纯看脸的话，模型觉得是什么
      - nlp_logits    : 纯看文字的话，模型觉得是什么
      - conflict_score: 0=完全一致, 1=严重冲突
      - alpha         : 模型最终更信 NLP(→1) 还是 CV(→0)
      - final_logits  : 综合判断
    """

    def __init__(
        self,
        cv_dim: int = CV_FEATURE_DIM,
        nlp_dim: int = NLP_FEATURE_DIM,
        num_classes: int = 4,  # 喜/怒/哀/讽
        gate_hidden: int = FUSION_GATE_HIDDEN,
        expert_hidden: int = EXPERT_HIDDEN_DIM,
        classifier_hidden: int = FUSION_HIDDEN_DIM,
    ):
        super(SarcasmAwareFusion, self).__init__()
        self.num_classes = num_classes
        concat_dim = cv_dim + nlp_dim

        # ---- 单模态专家 ----
        self.cv_expert = nn.Sequential(
            nn.Linear(cv_dim, expert_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(expert_hidden, num_classes),
        )
        self.nlp_expert = nn.Sequential(
            nn.Linear(nlp_dim, expert_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(expert_hidden, num_classes),
        )

        # ---- 冲突编码器 ----
        # 输入: cv_logits 和 nlp_logits 的差的绝对值 + 点积
        self.conflict_encoder = nn.Sequential(
            nn.Linear(num_classes * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),  # 0 = 一致, 1 = 冲突
        )

        # ---- 门控网络 ----
        self.gate = nn.Sequential(
            nn.Linear(concat_dim + num_classes * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # ---- 融合分类器 ----
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim + num_classes * 2, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden // 2, num_classes),
        )

    def forward(self, cv_feat, nlp_feat):
        """
        Returns:
            final_logits   : [B, num_classes]  融合后最终预测
            alpha          : [B, 1]            门控系数
            conflict_score : [B, 1]            冲突程度 (0~1)
            cv_logits      : [B, num_classes]  CV 单独看的结果
            nlp_logits     : [B, num_classes]  NLP 单独看的结果
        """
        B = cv_feat.size(0)

        # 1) 单模态专家各自预测
        cv_logits = self.cv_expert(cv_feat)        # [B, C]
        nlp_logits = self.nlp_expert(nlp_feat)     # [B, C]

        cv_probs = F.softmax(cv_logits, dim=1)
        nlp_probs = F.softmax(nlp_logits, dim=1)

        # 2) 冲突信号 —— 两个模态的预测分布差多少
        disagreement = torch.abs(cv_probs - nlp_probs)          # [B, C]
        agreement = cv_probs * nlp_probs                        # [B, C]  交集
        conflict_input = torch.cat([disagreement, agreement], dim=1)  # [B, 2C]
        conflict_score = self.conflict_encoder(conflict_input)  # [B, 1]

        # 3) 门控 —— 看到冲突后，调整信任策略
        raw_concat = torch.cat((cv_feat, nlp_feat), dim=1)
        gate_input = torch.cat([raw_concat, disagreement, agreement], dim=1)
        alpha = self.gate(gate_input)                   # [B, 1]

        # 4) 加权融合
        weighted_nlp = nlp_feat * alpha
        weighted_cv = cv_feat * (1.0 - alpha)
        fused_feat = torch.cat((weighted_cv, weighted_nlp), dim=1)

        # 5) 最终分类 —— 融合特征 + 冲突信号一起喂给分类器
        classifier_input = torch.cat([fused_feat, disagreement, agreement], dim=1)
        final_logits = self.classifier(classifier_input)

        return {
            "final_logits": final_logits,
            "alpha": alpha,
            "conflict_score": conflict_score,
            "cv_logits": cv_logits,
            "nlp_logits": nlp_logits,
        }

    # ---- 推理专用 ----
    @torch.inference_mode()
    def predict(self, cv_feat, nlp_feat):
        """单样本推理，返回可读的字典"""
        if cv_feat.dim() == 1:
            cv_feat = cv_feat.unsqueeze(0)
        if nlp_feat.dim() == 1:
            nlp_feat = nlp_feat.unsqueeze(0)

        out = self.forward(cv_feat, nlp_feat)

        final_probs = F.softmax(out["final_logits"], dim=1)[0]
        cv_probs = F.softmax(out["cv_logits"], dim=1)[0]
        nlp_probs = F.softmax(out["nlp_logits"], dim=1)[0]

        final_pred = torch.argmax(final_probs).item()
        cv_pred = torch.argmax(cv_probs).item()
        nlp_pred = torch.argmax(nlp_probs).item()

        return {
            "final": {
                "label": final_pred,
                "confidence": final_probs[final_pred].item(),
                "probabilities": final_probs.tolist(),
            },
            "cv_only": {
                "label": cv_pred,
                "confidence": cv_probs[cv_pred].item(),
                "probabilities": cv_probs.tolist(),
            },
            "nlp_only": {
                "label": nlp_pred,
                "confidence": nlp_probs[nlp_pred].item(),
                "probabilities": nlp_probs.tolist(),
            },
            "alpha": out["alpha"][0].item(),
            "conflict_score": out["conflict_score"][0].item(),
            "is_conflict": out["conflict_score"][0].item() > 0.5,
        }

    def analyze_conflict(self, cv_feat, nlp_feat, class_names=None):
        """
        专门分析冲突场景: CV 和 NLP 各看到什么，模型最终如何裁决。

        使用场景:
          >>> result = model.analyze_conflict(cv_feat, nlp_feat, ["喜","怒","哀","讽"])
          >>> print(result["summary"])
          "CV 认为 '喜' (0.81) | NLP 认为 '怒' (0.62) | 冲突高 (0.83) → 最终判断: '讽'"
        """
        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]

        r = self.predict(cv_feat, nlp_feat)

        lines = [
            f"CV  认为: '{class_names[r['cv_only']['label']]}' "
            f"(conf={r['cv_only']['confidence']:.2f})",

            f"NLP 认为: '{class_names[r['nlp_only']['label']]}' "
            f"(conf={r['nlp_only']['confidence']:.2f})",

            f"冲突程度: {r['conflict_score']:.2f} "
            f"{'⚠️ 高冲突' if r['is_conflict'] else '✓ 一致'}",

            f"门控 α = {r['alpha']:.2f} "
            f"(模型{'更信 NLP' if r['alpha'] > 0.5 else '更信 CV'})",

            f"→ 最终判断: '{class_names[r['final']['label']]}' "
            f"(conf={r['final']['confidence']:.2f})",
        ]

        return {
            **r,
            "summary": " | ".join(lines),
        }
