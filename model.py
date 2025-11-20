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
    
    def calculate(self, Q, K, V, mask):
        attn = torch.matmul(Q, torch.transpose(K, -1, -2))
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
    # def __init__(self, input_dim, latent_dim, enc_layers, dec_layers):
    def __init__(self, input_dim, latent_dim, enc_layers, dec_layers, name="Unknown"):  # 加
        super(VAE, self).__init__()
        self.name = name  # 保存 VAE 的名字（加）
        
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
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.decoder(z)
        reconstructed = self.fc_output(z)
        return reconstructed
    
    def reparameterize(self, mu, logvar):
        # ------------------ 【核心修复 1】: 钳制 logvar 的幅度 ------------------
        # 这是防止 std 瞬间指数爆炸到 Inf 的关键防线。
        # 限制 logvar 在 -20 到 20 之间，确保 exp(0.5 * logvar) 不会太大。
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        # --------------------------------------------------------------------
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # ------------------ 【Z 向量数值检查 - 调试代码】 ------------------
        # 使用 getattr 安全获取名称，避免 [Unknown] 导致崩溃
        display_name = getattr(self, 'name', 'VAE (Naming Failed)')
        
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"🚨🚨🚨 ALERT: VAE [{display_name}] 潜变量 Z 包含 NaN/Inf 🚨🚨🚨")
            print(f"  Z (max/min): {z.max().item():.2f} / {z.min().item():.2f}")
            print(f"  mu (max/min): {mu.max().item():.2f} / {mu.min().item():.2f}")
            print(f"  logvar (max/min): {logvar.max().item():.2f} / {logvar.min().item():.2f}")
            variance = std.pow(2)
            print(f"  Variance (max/min): {variance.max().item():.2e} / {variance.min().item():.2e}")
        
        if z.max() > 1000 or z.min() < -1000:
            print(f"💥💥💥 ALERT: VAE [{display_name}] Z 幅度爆炸! Max: {z.max().item():.2f}")
        # --------------------------------------------------------------------
        
        # ------------------ 【核心修复 2】: 钳制 Z ------------------
        # 即使 logvar 被钳制，Z 也应该被钳制以防万一。
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        return z
    
    # def forward(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     reconstructed = self.decode(z)
    #     return reconstructed, mu, logvar, z
    
    def forward(self, x):
        
        # ------------------ 【鲁棒性检查 1】: 检查 VAE 输入 ------------------
        # 检查输入 x 是否已包含 NaN/Inf (排除脏数据源头)
        display_name = getattr(self, 'name', 'VAE (Input Failed)')
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"🛑🛑🛑 CRITICAL ALERT: VAE [{display_name}] INPUT CONTAINS NaN/INF!")
            # ⚠️ 注意: 如果输入数据脏了，这里应该执行跳过或返回 0 损失的逻辑。
            # 为了不破坏原有逻辑，我们只打印警告，让 nan 继续传播，但我们会钳制 mu/logvar。
        
        # VAE 编码器：将输入 x 映射到潜空间参数 mu 和 logvar
        mu, logvar = self.encode(x)
        
        # ------------------ 【核心修复 1】: 钳制 MLP 的直接输出 ------------------
        # 这是防止 MLP 权重导致 mu/logvar 瞬间爆炸到 Inf 的最后一道防线。
        # 限制在一个很大的安全范围内，以防 MLP 输出值域失控。
        mu = torch.clamp(mu, min=-1e3, max=1e3)
        logvar = torch.clamp(logvar, min=-1e3, max=1e3)
        # ----------------------------------------------------------------------
        
        # z 的计算：reparameterize 内部现在也包含 logvar 钳制（防止 log(0)）和 z 钳制（防止 z 爆炸）
        # 如果 nan/inf 在这里仍然出现，reparameterize 内部的调试代码会打印更详细的信息。
        z = self.reparameterize(mu, logvar)
        
        # 解码器
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar, z


class TMEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.ent_num = kgs.ent_num
        self.rel_num = kgs.rel_num
        self.kgs = kgs
        self.args = args
        
        # ------------------ 【新增 VAE 名称】 ------------------
        self.ir_vae = VAE(input_dim=self.args.dim, latent_dim=64, enc_layers=2, dec_layers=2, name="IR_VAE")
        self.ar_vae = VAE(input_dim=self.args.dim, latent_dim=64, enc_layers=2, dec_layers=2, name="AR_VAE")
        
        self.a_vae = VAE(input_dim=self.args.dim, latent_dim=64, enc_layers=2, dec_layers=2, name="A_VAE")
        self.i_vae = VAE(input_dim=self.args.dim, latent_dim=64, enc_layers=2, dec_layers=2, name="I_VAE")
        # ----------------------------------------------------
        
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
    
    def forward(self, p_h, p_r, p_t, n_h, n_r, n_t):
        r_p_h = self.r_rep(p_h)
        r_p_r = F.normalize(self.rel_embed(p_r), 2, -1)
        r_p_t = self.r_rep(p_t)
        
        r_n_h = self.r_rep(n_h)
        r_n_r = F.normalize(self.rel_embed(n_r), 2, -1)
        r_n_t = self.r_rep(n_t)
        
        pos_dis = r_p_h + r_p_r - r_p_t
        neg_dis = r_n_h + r_n_r - r_n_t
        
        pos_score = torch.sum(torch.square(pos_dis), dim=1)
        neg_score = torch.sum(torch.square(neg_dis), dim=1)
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
    
    def gene_loss(self, recon_output, original_output, mu, logvar):
        recon_loss = nn.MSELoss()(recon_output, original_output)
        
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # ------------------ 【重新启用修复 2】: 钳制方差防止 log(0) ------------------
        # 确保方差不会下溢到 0，导致 log(0) -> -Inf
        variance = logvar.exp().clamp(min=1e-8)
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - variance)
        #-----
        return recon_loss + kl_loss
    
    def miss_generation(self, e_r, e_i, e_a, a_mask, i_mask):
        with torch.no_grad():
            mask_row = a_mask * i_mask
            e_i_detach = e_i.detach()
            e_a_detach = e_a.detach()
            e_r_detach = e_r.detach()
            
        # ------------------ 【新增代码】: 检查原始输入 Embeddings ------------------
        # 打印原始嵌入的最大值，如果它是 Inf 或 NaN，就会在这里被标记
        if torch.isnan(e_r_detach).any() or torch.isinf(e_r_detach).any():
            print(f"🔴🔴🔴 CRITICAL: e_r (Relation Embeddings) is contaminated! Max: {e_r_detach.max().item():.2f}")
        if torch.isnan(e_i_detach).any() or torch.isinf(e_i_detach).any():
            print(f"🔴🔴🔴 CRITICAL: e_i (Image Embeddings) is contaminated! Max: {e_i_detach.max().item():.2f}")
        if torch.isnan(e_a_detach).any() or torch.isinf(e_a_detach).any():
            print(f"🔴🔴🔴 CRITICAL: e_a (Attribute Embeddings) is contaminated! Max: {e_a_detach.max().item():.2f}")
        # -------------------------------------------------------------------------
        
        # 检查是否有数据可以用于 VAE
        if e_r_detach.numel() == 0:
            return 0, e_a_detach, e_i_detach
        
        ir_input = self.fc_map_1(torch.cat((e_i_detach, e_r_detach), dim=-1))
        ar_input = self.fc_map_2(torch.cat((e_a_detach, e_r_detach), dim=-1))
        
        # ------------------ 【新增代码】: 钳制 FC Map 输出 ------------------
        # 钳制 FC 映射层的输出，防止它们在进入 VAE 之前就爆炸
        ir_input = torch.clamp(ir_input, min=-1e3, max=1e3)
        ar_input = torch.clamp(ar_input, min=-1e3, max=1e3)
        # ------------------------------------------------------------------
        
        # generate a with i and r
        gen_ir, ir_mu, ir_logvar, ir_latent = self.ir_vae(ir_input)
        gen_a, a_mu, a_logvar, a_latent = self.a_vae(e_a_detach)
        
        comp_a = self.a_vae.decode(ir_latent)
        # generate i with a and r
        gen_ar, ar_mu, ar_logvar, ar_latent = self.ar_vae(ar_input)
        gen_i, i_mu, i_logvar, i_latent = self.i_vae(e_i_detach)
        
        comp_i = self.i_vae.decode(ar_latent)
        # optimize
        mmd_loss = self.gene_loss(gen_a[mask_row.bool()], e_a_detach[mask_row.bool()], a_mu[mask_row.bool()],
                                  a_logvar[mask_row.bool()]) + self.gene_loss(gen_ir[mask_row.bool()],
                                                                              ir_input[mask_row.bool()],
                                                                              ir_mu[mask_row.bool()], ir_logvar[
                                                                                  mask_row.bool()]) + self.gene_loss(
            gen_i[mask_row.bool()], e_i_detach[mask_row.bool()], i_mu[mask_row.bool()],
            i_logvar[mask_row.bool()]) + self.gene_loss(gen_ar[mask_row.bool()], ar_input[mask_row.bool()],
                                                        ar_mu[mask_row.bool()],
                                                        ar_logvar[mask_row.bool()]) + self.mse_factor * nn.MSELoss()(
            a_latent[mask_row.bool()], ir_latent[mask_row.bool()]) + self.mse_factor * nn.MSELoss()(
            i_latent[mask_row.bool()], ar_latent[mask_row.bool()])
        
        e_i_comp = torch.where(i_mask.unsqueeze(-1).bool(), e_i, comp_i)
        e_a_comp = torch.where(a_mask.unsqueeze(-1).bool(), e_a, comp_a)
        
        # ------------------ 【最终核心修复】: 钳制 TMEA 的所有参数值 ------------------
        # 钳制所有需要梯度的参数的 *值* (param.data)，确保没有一个参数值逃逸到 NaN
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 限制在 -10.0 到 10.0 之间，这是安全的嵌入值域。
                param.data = torch.clamp(param.data, min=-10.0, max=10.0)
        # ----------------------------------------------------------------------------
        
        return mmd_loss, e_a_comp, e_i_comp
    
    def cross_attention(self, a, b, c):
        # a=R (Relation), b=I (Image), c=A (Attribute)
        
        is_b_present = b.numel() > 0
        batch_size = a.size(0)
        
        # 修复：确保 self.args.dim 是 TMEA 的正确属性
        zero_embed = torch.zeros(batch_size, self.args.dim, device=a.device)
        
        w_normalized = F.softmax(self.modal_weight, dim=-1)
        
        # Cross Attention 1: Query b (I), K/V a (R)
        ab = self.ca_ab(b, a) if is_b_present else zero_embed
        
        # Cross Attention 2: Query c (A), K/V a (R)
        ac = self.ca_ac(c, a)
        
        # Alignment for a (R)
        a_align = w_normalized[1, 0] * a + w_normalized[1, 1] * ab + w_normalized[1, 2] * ac
        
        # Cross Attention 3: Query a (R), K/V b (I)
        ba = self.ca_ba(a, b) if is_b_present else zero_embed
        
        # Cross Attention 4: Query c (A), K/V b (I)
        bc = self.ca_bc(c, b) if is_b_present else zero_embed
        
        # Alignment for b (I)
        if is_b_present:
            b_align = w_normalized[2, 0] * b + w_normalized[2, 1] * ba + w_normalized[2, 2] * bc
        else:
            b_align = zero_embed
            
            # Cross Attention 5: Query a (R), K/V c (A)
        ca = self.ca_ca(a, c)
        
        # Cross Attention 6: Query b (I), K/V c (A)
        cb = self.ca_cb(b, c) if is_b_present else zero_embed
        
        # Alignment for c (A)
        c_align = w_normalized[3, 0] * c + w_normalized[3, 1] * ca + w_normalized[3, 2] * cb
        
        # Joint Embedding
        joint_emb = torch.cat([a_align, b_align, c_align], dim=1)
        
        # Orthogonal Loss (仅计算非零的部分)
        if is_b_present:
            orth_loss = self.orth_loss(b, a - ab) + self.orth_loss(c, a - ac) + self.orth_loss(a,
                                                                                               b - ba) + self.orth_loss(
                c, b - bc) + self.orth_loss(a, c - ca) + self.orth_loss(b, c - cb)
        else:
            orth_loss = self.orth_loss(c, a - ac) + self.orth_loss(a, c - ca)
        
        orth_loss = self.orth_factor * orth_loss
        
        return joint_emb, orth_loss
    
    def orth_loss(self, x, y):
        orth = torch.mean(x * y, dim=-1)
        loss = torch.mean(torch.pow(orth, 2))
        return loss
    