import os
import json
import torch
import numpy as np
import h5py
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from collections import defaultdict

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "FB15K-DB15K")

# BERT 模型本地路径
BERT_LOCAL_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "bert-base-uncased")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FEATURE_DIM = 768  # BERT-base 的输出维度
# ===============================================

print(f"Using Device: {DEVICE}")
print(f"Local BERT Model Directory: {BERT_LOCAL_MODEL_DIR}")


def load_bert_model():
    """尝试从本地加载 BERT 模型"""
    try:
        if os.path.exists(os.path.join(BERT_LOCAL_MODEL_DIR, "pytorch_model.bin")):
            print("Loading BERT model from local files...")
            tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_MODEL_DIR)
            model = BertModel.from_pretrained(BERT_LOCAL_MODEL_DIR).to(DEVICE)
            print("Successfully loaded BERT model from local path.")
            return tokenizer, model
        else:
            raise FileNotFoundError("pytorch_model.bin not found in local BERT directory.")
    except Exception as e:
        print("\n" + "=" * 80)
        print("CRITICAL ERROR: Failed to load BERT model from local path.")
        print(f"Underlying Exception: {e}")
        print(
            f"Please check if {BERT_LOCAL_MODEL_DIR} contains config.json, vocab.txt, and the 440MB pytorch_model.bin.")
        print("=" * 80 + "\n")
        raise


def extract_unique_attributes(triple_file_path):
    """从数值三元组文件中提取所有唯一的属性名称"""
    unique_attrs = set()
    print(f"Extracting unique attributes from: {triple_file_path}")
    
    try:
        with open(triple_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                # 使用 | 作为分隔符，因为它在 FB15K-DB15K 数据集中更常见 (尽管 kgs.py 用 \t，我们以最保守的方式处理)
                params = line.strip().split('\t')
                if len(params) == 3:
                    unique_attrs.add(params[1].strip())
    except FileNotFoundError:
        print(f"Warning: {triple_file_path} not found. Returning empty set.")
    except Exception as e:
        print(f"Error reading triple file {triple_file_path}: {e}")
    
    # 移除可能存在的空字符串
    unique_attrs.discard('')
    print(f"Found {len(unique_attrs)} unique attributes.")
    return unique_attrs


def generate_attribute_embeddings(tokenizer, model, triple_file1, triple_file2, output_h5_1, output_h5_2):
    """生成属性嵌入并保存到 HDF5 文件"""
    
    attr_f1 = os.path.join(DATA_DIR, triple_file1)
    attr_f2 = os.path.join(DATA_DIR, triple_file2)
    
    # 1. 提取所有属性
    attrs1 = extract_unique_attributes(attr_f1)
    attrs2 = extract_unique_attributes(attr_f2)
    
    # 2. 生成并保存第一个KG的属性嵌入
    save_embeddings(tokenizer, model, attrs1, os.path.join(DATA_DIR, output_h5_1))
    
    # 3. 生成并保存第二个KG的属性嵌入
    save_embeddings(tokenizer, model, attrs2, os.path.join(DATA_DIR, output_h5_2))


def save_embeddings(tokenizer, model, attributes_set, output_h5_path):
    """编码并保存属性到HDF5"""
    if not attributes_set:
        print(f"No attributes to process for {output_h5_path}. Creating empty file.")
        with h5py.File(output_h5_path, 'w') as f:
            pass
        return
    
    print(f"\nEncoding {len(attributes_set)} attributes for {os.path.basename(output_h5_path)}...")
    
    # 属性编码和保存
    with h5py.File(output_h5_path, 'w') as hf:
        model.eval()
        for attr_name in tqdm(attributes_set):
            # 标记化和编码
            inputs = tokenizer(attr_name, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            
            with torch.no_grad():
                # 提取 [CLS] token 的嵌入作为属性特征
                outputs = model(**inputs)
                # outputs[0] 是最后一层的隐藏状态，outputs[0][:, 0, :] 是 [CLS] 向量
                embedding = outputs[0][:, 0, :].squeeze().cpu().numpy()
                
                # 保存到 HDF5 文件，键为属性名称
            hf.create_dataset(attr_name, data=embedding)
    
    print(f"Successfully saved {len(attributes_set)} attribute embeddings to {output_h5_path}")


if __name__ == "__main__":
    try:
        # 步骤 1: 加载 BERT 模型
        tokenizer, model = load_bert_model()
        
        # 步骤 2: 提取并保存属性嵌入
        generate_attribute_embeddings(
            tokenizer,
            model,
            "FB15K_NumericalTriples.txt",
            "DB15K_NumericalTriples.txt",
            "attr_name_1.h5",
            "attr_name_2.h5"
        )
        
        # 步骤 3: 确保 kgs.py 需要的 bert_attr.pkl 仍然存在 (我们不需要修改它，因为它只是占位符)
        
        print("\n\nAll attribute feature extraction steps are complete. You can now run python main.py")
    
    except Exception:
        print("\nProcess failed. Please check the error message above for details.")