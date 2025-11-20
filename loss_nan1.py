# -----------------------------------------------------------------------
# 文件: loss.py (完整内容)
# -----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, score, label):
        """
        Calculates the Contrastive Loss for Entity Alignment.
        score: Similarity matrix (e.g., cosine similarity from torch.mm).
        label: Ground truth matrix (1 for positive pairs, 0 for negative pairs).
        """
        
        # 1. Calculate squared Euclidean distance: d_sq = 2 * (1 - score)
        dis_sq = torch.clamp(2.0 * (1.0 - score), min=0)
        
        # 2. Calculate un-squared Euclidean distance: d = sqrt(d_sq)
        dis = torch.sqrt(dis_sq)
        
        # 3. 计算正样本损失 (Positive Loss):
        pos_loss = label * dis_sq
        
        # 4. 计算负样本损失 (Negative Loss):
        neg_loss = (1 - label) * torch.pow(torch.clamp(self.margin - dis, min=0.0), 2)
        
        # 5. 平均损失
        loss_contrastive = torch.mean(pos_loss + neg_loss)
        return loss_contrastive


# ====================================================================
# 【核心修正】: 添加数值稳定的 MMD RBF 核函数和 MMD 损失计算
# --------------------------------------------------------------------

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算高斯核矩阵
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    # 计算所有样本之间的 L2 距离平方
    total_sq = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    L2_distance_sq = ((total_sq - total.unsqueeze(1).expand_as(total_sq)) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # 使用中值启发式方法估算带宽
        # 【修正点】: 添加 1e-8 极小值保护，防止除以零或数值不稳定
        bandwidth = torch.sum(L2_distance_sq.data) / (n_samples ** 2 - n_samples) + 1e-8
    
    # 多核设置
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    # 计算核矩阵
    kernel_val = [torch.exp(-L2_distance_sq / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
    
    def forward(self, source, target):
        '''
        计算无偏 MMD 损失 (Maximum Mean Discrepancy)
        '''
        n_s = source.size(0)
        n_t = target.size(0)
        
        # 【修正点】: 防止空输入导致 nan
        if n_s == 0 or n_t == 0:
            return torch.tensor(0.0, device=source.device)
        
        # 【修正点】: L2 归一化输入，防止核函数计算时数值溢出
        source = F.normalize(source, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        
        kernels = guassian_kernel(source, target,
                                  kernel_mul=self.kernel_mul,
                                  kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        
        # 提取各个部分
        XX = kernels[:n_s, :n_s]
        YY = kernels[n_s:, n_s:]
        XY = kernels[:n_s, n_s:]
        
        # 无偏估计 (Unbiased estimate)
        loss = (torch.sum(XX) - torch.trace(XX)) / (n_s * (n_s - 1))
        loss += (torch.sum(YY) - torch.trace(YY)) / (n_t * (n_t - 1))
        loss -= 2 * torch.sum(XY) / (n_s * n_t)
        
        return loss