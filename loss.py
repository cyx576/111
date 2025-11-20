import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        # Margin should be a float or a torch tensor.
        self.margin = margin
    
    def forward(self, score, label):
        """
        Calculates the Contrastive Loss for Entity Alignment.
        score: Similarity matrix (e.g., cosine similarity from torch.mm).
        label: Ground truth matrix (1 for positive pairs, 0 for negative pairs).
        """
        
        # 1. Calculate squared Euclidean distance
        # 建议在这里就加上 1e-9，或者在 sqrt 里面加
        dis_sq = torch.clamp(2.0 * (1.0 - score), min=0)
        
        # 2. 【修复】: 加上 1e-9 防止对 0 开根号导致梯度爆炸
        dis = torch.sqrt(dis_sq + 1e-9)
        
        # 3. 计算正样本损失 (Positive Loss): 
        # L_pos = label * d^2 (我们希望正样本距离小)
        pos_loss = label * dis_sq
        
        # 4. 计算负样本损失 (Negative Loss): 
        # L_neg = (1 - label) * max(0, margin - d)^2 (我们希望负样本距离大于 margin)
        neg_term = torch.clamp(self.margin - dis, min=0)
        # 使用 torch.square 代替 torch.pow(tensor, 2) 更简洁，且能确保是 Tensor 操作。
        neg_loss = (1.0 - label) * torch.square(neg_term)
        
        # Total Loss: 平均所有损失项
        loss_contrastive = torch.mean(pos_loss + neg_loss)
        return loss_contrastive