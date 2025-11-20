from math import e
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import time
import sys
from utils import *
from evaluation import *


class CrossAttention(nn.Module):
    def __init__(self, args, head=2):
        super().__init__()
        self.hidden_size = args.dim
        self.head = head
        self.h_size = self.hidden_size // self.head
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_output = nn.Linear(self.hidden_size, self.hidden_size)
        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_k.weight)
        nn.init.xavier_normal_(self.linear_v.weight)
        nn.init.xavier_normal_(self.linear_output.weight)
    
    # def calculate(self, Q, K, V, mask):
    #     attn = torch.matmul(Q, torch.transpose(K,-1,-2))
    #     if mask is not None: attn = attn.masked_fill(mask, -1e9)
    #     attn = torch.softmax(attn / (Q.size(-1) ** 0.5), dim=-1)
    #     attn = torch.matmul(attn, V)
    #     return attn
    
    def calculate(self, Q, K, V, mask):
        attn = torch.matmul(Q, torch.transpose(K, -1, -2))
        
        # 【最终核心修复】: 强制裁剪注意力分数
        # 这是防止 Cross-Attention 梯度爆炸的唯一方法
        attn = torch.clamp(attn, min=-50.0, max=50.0)
        
        if mask is not None: attn = attn.masked_fill(mask, -1e9)
        attn = torch.softmax(attn / (Q.size(-1) ** 0.5), dim=-1)
        attn = torch.matmul(attn, V)
        return attn
    
    def forward(self, x, y, attention_mask=None):
        batch_size = x.size(0)
        
        # 修复：检查 batch_size 是否为 0
        if batch_size == 0:
            return torch.empty(0, self.hidden_size, device=x.device)
        
        q_s = self.linear_q(x).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
        k_s = self.linear_k(y).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
        v_s = self.linear_v(y).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
        if attention_mask is not None: attention_mask = attention_mask.eq(0)
        attn = self.calculate(q_s, k_s, v_s, attention_mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, self.hidden_size)
        attn = self.linear_output(attn)
        return attn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_layers, dec_layers):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        
        # encoder
        enc_modules = []
        for i in range(enc_layers):
            if i == 0:
                enc_modules.append(nn.Linear(input_dim, self.latent_dim))
            else:
                enc_modules.append(nn.Linear(self.latent_dim, self.latent_dim))
            enc_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_modules)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        
        # decoder
        dec_modules = []
        for i in range(dec_layers):
            if i == 0:
                dec_modules.append(nn.Linear(self.latent_dim, self.latent_dim))
            else:
                dec_modules.append(nn.Linear(self.latent_dim, self.latent_dim))
            dec_modules.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_modules)
        self.fc_output = nn.Linear(self.latent_dim, input_dim)
        
        self.init_weights()
    
    def init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
        
        nn.init.xavier_normal_(self.fc_mu.weight.data)
        nn.init.xavier_normal_(self.fc_logvar.weight.data)
        nn.init.xavier_normal_(self.fc_output.weight.data)
    
    # def encode(self, x):
    #     x = self.encoder(x)
    #     mu = self.fc_mu(x)
    #     logvar = self.fc_logvar(x)
    #     return mu, logvar
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # 【核心修复 1】: 强制裁剪 VAE 的 Mu 和 LogVar
        # 防止 mu 过大导致 latent code 爆炸，也防止 logvar 导致 exp(logvar) 过大
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-5.0, max=5.0)
        
        return mu, logvar
    
    def decode(self, z):
        z = self.decoder(z)
        reconstructed = self.fc_output(z)
        return reconstructed
    
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     z = mu + eps * std
    #     return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 【核心修复 1】: 强制裁剪潜变量
        # 潜变量 Z 限制在安全范围内，防止其 L2 范数在 MMD/MSE 中爆炸
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z


class TMEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.ent_num = kgs.ent_num
        self.rel_num = kgs.rel_num
        self.kgs = kgs
        self.args = args
        self.hidden_size = args.dim
        self.modal_weight = nn.Parameter(
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
        self.img_embed = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(kgs.images_list)))
        self.atr_embed = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(kgs.attr_emb_list)))
        self.ent_embed = nn.Embedding(self.ent_num, self.args.dim)
        self.rel_embed = nn.Embedding(self.rel_num, self.args.dim)
        nn.init.xavier_normal_(self.ent_embed.weight.data)
        nn.init.xavier_normal_(self.rel_embed.weight.data)
        self.fc_i = nn.Linear(768, self.args.dim)
        self.fc_a = nn.Linear(768, self.args.dim)
        nn.init.xavier_normal_(self.fc_i.weight.data)
        nn.init.xavier_normal_(self.fc_a.weight.data)
        
        self.fc_map_1 = nn.Linear(2 * self.args.dim, self.args.dim)
        self.fc_map_2 = nn.Linear(2 * self.args.dim, self.args.dim)
        
        nn.init.xavier_normal_(self.fc_map_1.weight.data)
        nn.init.xavier_normal_(self.fc_map_2.weight.data)
        
        self.ca_ab = CrossAttention(self.args)
        self.ca_ac = CrossAttention(self.args)
        self.ca_bc = CrossAttention(self.args)
        self.ca_ba = CrossAttention(self.args)
        self.ca_ca = CrossAttention(self.args)
        self.ca_cb = CrossAttention(self.args)
        self.orth_factor = self.args.orth_factor
        self.mse_factor = self.args.mse_factor
        
        self.ir_vae = VAE(input_dim=100, latent_dim=64, enc_layers=2, dec_layers=2)
        self.ar_vae = VAE(input_dim=100, latent_dim=64, enc_layers=2, dec_layers=2)
        
        self.a_vae = VAE(input_dim=100, latent_dim=64, enc_layers=2, dec_layers=2)
        self.i_vae = VAE(input_dim=100, latent_dim=64, enc_layers=2, dec_layers=2)
    
    # def forward(self, x, y, attention_mask=None):
    #     batch_size = x.size(0)
    #     q_s = self.linear_q(x).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
    #     k_s = self.linear_k(y).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
    #     v_s = self.linear_v(y).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
    #     if attention_mask is not None: attention_mask = attention_mask.eq(0)
    #     attn = self.calculate(q_s, k_s, v_s, attention_mask)
    #     attn = attn.transpose(1, 2).contiguous().view(batch_size, self.hidden_size)
    #     attn = self.linear_output(attn)
    #     return attn
    
    def forward(self, p_h, p_r, p_t, n_h, n_r, n_t):
        # 接收 6 个参数：正样本 (p) 和 负样本 (n) 的 h, r, t
        
        # 获取归一化后的实体和关系表示
        r_p_h = self.r_rep(p_h)
        r_p_r = F.normalize(self.rel_embed(p_r), 2, -1)
        r_p_t = self.r_rep(p_t)
        
        r_n_h = self.r_rep(n_h)
        r_n_r = F.normalize(self.rel_embed(n_r), 2, -1)
        r_n_t = self.r_rep(n_t)
        
        # 计算距离 (TransE score: h + r - t)
        pos_dis = r_p_h + r_p_r - r_p_t
        neg_dis = r_n_h + r_n_r - r_n_t
        
        # 计算分数的平方和
        pos_score = torch.sum(torch.square(pos_dis), dim=1)
        neg_score = torch.sum(torch.square(neg_dis), dim=1)
        
        # 计算 Margin Ranking Loss
        # 这里的 margin 来自 args
        relation_loss = torch.sum(F.relu(self.args.margin + pos_score - neg_score))
        
        return relation_loss
    
    def predict(self, e1, e2, e1_mask, e1_im_mask, e2_mask, e2_im_mask, mode='test'):
        
        # 兜底检查：如果输入实体列表为空，直接返回空结果，避免后续张量操作崩溃。
        if e1.numel() == 0 or e2.numel() == 0:
            dim = self.args.dim
            # 修复：确保返回的是 PyTorch 张量 (Tensor)，而不是 Python 整数 0
            # 使用输入的设备创建零张量，以防止 Type Error
            zero_tensor = torch.tensor(0.0, device=e1.device)
            if mode == 'test':
                # 返回空的 numpy 数组，确保形状正确 (0, dim)
                empty_r = np.empty((0, dim))
                empty_all = np.empty((0, 3 * dim))
                return empty_all, empty_all, empty_r, empty_r, empty_r, empty_r, empty_r, empty_r
            else:
                # Returns: r_score, a_score, i_score, score, orth_loss, mmd_loss
                return zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor
        
        e1_r_embed = self.r_rep(e1)
        e2_r_embed = self.r_rep(e2)
        e1_i_embed = self.i_rep(e1)
        e2_i_embed = self.i_rep(e2)
        
        e1_a_embed = self.a_rep(e1)
        e2_a_embed = self.a_rep(e2)
        
        mmd_loss1, e1_a_comp, e1_i_comp = self.miss_generation(e1_r_embed, e1_i_embed, e1_a_embed, e1_mask, e1_im_mask)
        mmd_loss2, e2_a_comp, e2_i_comp = self.miss_generation(e2_r_embed, e2_i_embed, e2_a_embed, e2_mask, e2_im_mask)
        
        e1_all, orth_loss1 = self.cross_attention(e1_r_embed, e1_i_comp, e1_a_comp)
        e2_all, orth_loss2 = self.cross_attention(e2_r_embed, e2_i_comp, e2_a_comp)
        
        if mode == 'test':
            return e1_all.cpu().numpy(), e2_all.cpu().numpy(), \
                e1_r_embed.cpu().numpy(), e2_r_embed.cpu().numpy(), \
                e1_i_embed.cpu().numpy(), e2_i_embed.cpu().numpy(), \
                e1_a_embed.cpu().numpy(), e2_a_embed.cpu().numpy()
        else:
            r_score = torch.mm(e1_r_embed, e2_r_embed.t())
            a_score = torch.mm(e1_a_embed, e2_a_embed.t())
            i_score = torch.mm(e1_i_embed, e2_i_embed.t())
            score = torch.mm(e1_all, e2_all.t())
            if mode == 'train':
                return r_score, a_score, i_score, score, orth_loss1 + orth_loss2, mmd_loss1 + mmd_loss2
            else:
                return r_score, a_score, i_score, score
    
    def r_rep(self, e):
        return F.normalize(self.ent_embed(e), 2, -1)
    
    def i_rep(self, e):
        return F.normalize(self.fc_i(self.img_embed(e)), 2, -1)
    
    def a_rep(self, e):
        return F.normalize(self.fc_a(self.atr_embed(e)), 2, -1)
    
    # def gene_loss(self, recon_output, original_output, mu, logvar):
    #     recon_loss = nn.MSELoss()(recon_output, original_output)
    #     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     return recon_loss + kl_loss
    
    def gene_loss(self, recon_output, original_output, mu, logvar):
        recon_loss = nn.MSELoss()(recon_output, original_output)
        
        # 【核心修复 2】: VAE KL 散度数值稳定
        # 裁剪方差的指数，防止其接近 0，确保分母不为 0
        variance = logvar.exp().clamp(min=1e-8)
        
        # 修正后的 KL 散度计算 (使用裁剪后的 variance)
        kl_loss_per_element = 1 + logvar - mu.pow(2) - variance
        kl_loss = -0.5 * torch.sum(kl_loss_per_element)
        
        return recon_loss + kl_loss
    
    # def miss_generation(self, e_r, e_i, e_a, a_mask, i_mask):
    #     with torch.no_grad():
    #         mask_row = a_mask * i_mask
    #         e_i_detach = e_i.detach()
    #         e_a_detach = e_a.detach()
    #         e_r_detach = e_r.detach()
    #
    #     ir_input = self.fc_map_1(torch.cat((e_i_detach, e_r_detach), dim=-1))
    #     ar_input = self.fc_map_2(torch.cat((e_a_detach, e_r_detach), dim=-1))
    #
    #     # generate a with i and r
    #     gen_ir, ir_mu, ir_logvar, ir_latent = self.ir_vae(ir_input)
    #     gen_a, a_mu, a_logvar, a_latent = self.a_vae(e_a_detach)
    #
    #     comp_a = self.a_vae.decode(ir_latent)
    #     # generate i with a and r
    #     gen_ar, ar_mu, ar_logvar, ar_latent = self.ar_vae(ar_input)
    #     gen_i, i_mu, i_logvar, i_latent = self.i_vae(e_i_detach)
    #
    #     comp_i = self.i_vae.decode(ar_latent)
    #     # optimize
    #     mmd_loss = self.gene_loss(gen_a[mask_row.bool()], e_a_detach[mask_row.bool()], a_mu[mask_row.bool()],
    #                               a_logvar[mask_row.bool()]) + self.gene_loss(gen_ir[mask_row.bool()],
    #                                                                           ir_input[mask_row.bool()],
    #                                                                           ir_mu[mask_row.bool()], ir_logvar[
    #                                                                               mask_row.bool()]) + self.gene_loss(
    #         gen_i[mask_row.bool()], e_i_detach[mask_row.bool()], i_mu[mask_row.bool()],
    #         i_logvar[mask_row.bool()]) + self.gene_loss(gen_ar[mask_row.bool()], ar_input[mask_row.bool()],
    #                                                     ar_mu[mask_row.bool()],
    #                                                     ar_logvar[mask_row.bool()]) + self.mse_factor * nn.MSELoss()(
    #         a_latent[mask_row.bool()], ir_latent[mask_row.bool()]) + self.mse_factor * nn.MSELoss()(
    #         i_latent[mask_row.bool()], ar_latent[mask_row.bool()])
    #
    #     e_i_comp = torch.where(i_mask.unsqueeze(-1).bool(), e_i, comp_i)
    #     e_a_comp = torch.where(a_mask.unsqueeze(-1).bool(), e_a, comp_a)
    #     return mmd_loss, e_a_comp, e_i_comp
    
    def miss_generation(self, e_r, e_i, e_a, a_mask, i_mask):
        with torch.no_grad():
            mask_row = a_mask * i_mask
            
            # 【修复 1: 关键】: 解决 mmd_loss 瞬间 nan 的问题 (空数据导致的 0/0)
            # 如果当前批次中没有需要计算 MMD 的实体（交集为空），直接返回 0 损失。
            if mask_row.sum() == 0:
                # 返回 0 损失和原始嵌入，确保梯度流程不中断
                return torch.tensor(0.0, device=e_r.device), e_a, e_i
            
            e_i_detach = e_i.detach()
            e_a_detach = e_a.detach()
            e_r_detach = e_r.detach()
        
        # VAE Input (已移除 F.normalize，保留了原始的 fc_map 输出)
        ir_input = self.fc_map_1(torch.cat((e_i_detach, e_r_detach), dim=-1))
        ar_input = self.fc_map_2(torch.cat((e_a_detach, e_r_detach), dim=-1))
        
        # generate a with i and r
        gen_ir, ir_mu, ir_logvar, ir_latent = self.ir_vae(ir_input)
        gen_a, a_mu, a_logvar, a_latent = self.a_vae(e_a_detach)
        
        # VAE Output 1
        comp_a = self.a_vae.decode(ir_latent)
        # 【修复 2】: VAE 输出的 L2 归一化 (防止 comp_a 幅度过大，导致 r_loss 爆炸)
        comp_a = F.normalize(comp_a, p=2, dim=-1)
        
        # generate i with a and r
        gen_ar, ar_mu, ar_logvar, ar_latent = self.ar_vae(ar_input)
        gen_i, i_mu, i_logvar, i_latent = self.i_vae(e_i_detach)
        
        # VAE Output 2
        comp_i = self.i_vae.decode(ar_latent)
        # 【修复 2】: VAE 输出的 L2 归一化 (防止 comp_i 幅度过大，导致 r_loss 爆炸)
        comp_i = F.normalize(comp_i, p=2, dim=-1)
        
        # optimize
        valid_mask = mask_row.bool()
        
        # MMD Loss 计算 (使用原始的 latent code 和损失计算，避免过度归一化导致的损失坍塌)
        # 注意：此处必须使用 valid_mask.sum() > 0 检查，以防出现浮点错误
        if valid_mask.sum() > 0:
            mmd_loss = self.gene_loss(gen_a[valid_mask], e_a_detach[valid_mask], a_mu[valid_mask],
                                      a_logvar[valid_mask]) + self.gene_loss(gen_ir[valid_mask],
                                                                             ir_input[valid_mask],
                                                                             ir_mu[valid_mask], ir_logvar[
                                                                                 valid_mask]) + self.gene_loss(
                gen_i[valid_mask], e_i_detach[valid_mask], i_mu[valid_mask],
                i_logvar[valid_mask]) + self.gene_loss(gen_ar[valid_mask], ar_input[valid_mask],
                                                       ar_mu[valid_mask],
                                                       ar_logvar[valid_mask]) + self.mse_factor * nn.MSELoss()(
                a_latent[valid_mask], ir_latent[valid_mask]) + self.mse_factor * nn.MSELoss()(
                i_latent[valid_mask], ar_latent[valid_mask])
        else:
            mmd_loss = torch.tensor(0.0, device=e_r.device)  # 理论上前面检查了，但这里是双重保险
        
        e_i_comp = torch.where(i_mask.unsqueeze(-1).bool(), e_i, comp_i)
        e_a_comp = torch.where(a_mask.unsqueeze(-1).bool(), e_a, comp_a)
        
        return mmd_loss, e_a_comp, e_i_comp
    
    def cross_attention(self, a, b, c):
        w_normalized = F.softmax(self.modal_weight, dim=-1)
        ab, ac = self.ca_ab(b, a), self.ca_ac(c, a)
        a_align = w_normalized[1, 0] * a + w_normalized[1, 1] * ab + w_normalized[1, 2] * ac
        ba, bc = self.ca_ba(a, b), self.ca_bc(c, b)
        b_align = w_normalized[2, 0] * b + w_normalized[2, 1] * ba + w_normalized[2, 2] * bc
        ca, cb = self.ca_ca(a, c), self.ca_cb(b, c)
        c_align = w_normalized[3, 0] * c + w_normalized[3, 1] * ca + w_normalized[3, 2] * cb
        
        # ---------【核心修复】: 稳定最终的对齐嵌入
        # 1. 钳制：限制最大幅度，防止 Contrastive Loss 平方爆炸
        a_align = torch.clamp(a_align, min=-10.0, max=10.0)
        b_align = torch.clamp(b_align, min=-10.0, max=10.0)
        c_align = torch.clamp(c_align, min=-10.0, max=10.0)
        
        # 2. 归一化：将特征约束在单位球上，确保 Contrastive Loss 距离计算稳定
        a_align = F.normalize(a_align, p=2, dim=-1)
        b_align = F.normalize(b_align, p=2, dim=-1)
        c_align = F.normalize(c_align, p=2, dim=-1)
        # --------------------
        
        joint_emb = torch.cat([a_align, b_align, c_align], dim=1)
        
        orth_loss = self.orth_loss(b, a - ab) + self.orth_loss(c, a - ac) + self.orth_loss(a, b - ba) + self.orth_loss(
            c, b - bc) + self.orth_loss(a, c - ca) + self.orth_loss(b, c - cb)
        orth_loss = self.orth_factor * orth_loss
        
        return joint_emb, orth_loss
    
    def orth_loss(self, x, y):
        orth = torch.mean(x * y, dim=-1)
        loss = torch.mean(torch.pow(orth, 2))
        return loss