import os
import re
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# ================= 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "FB15K-DB15K")
BERT_PATH = os.path.join(SCRIPT_DIR, "models", "bert-base-uncased")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ================= 健壮的实体读取函数 (与 kgs.py 保持一致) =================
def read_entities(filename):
    print(f"Reading entities from: {filename}")
    entities = set()
    is_dbpedia = 'DB15K' in os.path.basename(filename)
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            h, r, t = None, None, None
            
            if not is_dbpedia:  # FB15K
                if line.endswith('.'): line = line[:-1].strip()
                params = line.split()
                if len(params) == 3:
                    h, r, t = params
            else:  # DB15K
                match = re.search(r'<(.*?)>\s+<(.*?)>\s+<(.*?)>\s*\.?', line)
                if match:
                    h, r, t = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            
            if h and t:
                # 清理尖括号
                if h.startswith('<') and h.endswith('>'): h = h[1:-1]
                if t.startswith('<') and t.endswith('>'): t = t[1:-1]
                entities.add(h)
                entities.add(t)
    
    print(f"Found {len(entities)} unique entities.")
    return list(entities)


# ================= BERT 编码函数 =================
def generate_embeddings(entity_list, output_file, tokenizer, model):
    print(f"Generating embeddings for {output_file}...")
    data_dict = {}
    
    batch_size = 32
    model.eval()
    
    # 准备批次
    batches = [entity_list[i:i + batch_size] for i in range(0, len(entity_list), batch_size)]
    
    with torch.no_grad():
        for batch in tqdm(batches):
            # 准备输入文本
            texts = []
            for ent in batch:
                # 策略：如果是 DBpedia URI，提取名称；如果是 FB15K ID，直接用 ID (因为没有名称文件)
                if 'http' in ent:
                    match = re.search(r'/([^/]+)$', ent)
                    text = match.group(1).strip().replace('_', ' ') if match else ent
                else:
                    text = ent  # FB15K ID (e.g., /m/027rn)
                texts.append(text)
            
            # 编码
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32).to(DEVICE)
            outputs = model(**inputs)
            # 使用 [CLS] token (index 0)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 保存：Key 必须与 kgs.py 读取到的实体 ID 完全一致
            for i, ent in enumerate(batch):
                # 注意：kgs.py 读取特征时期望的是 [embedding] 列表形式
                data_dict[ent] = [embeddings[i]]
    
    print(f"Saving {len(data_dict)} embeddings to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)


# ================= 主流程 =================
if __name__ == "__main__":
    # 1. 加载 BERT
    if not os.path.exists(os.path.join(BERT_PATH, "pytorch_model.bin")):
        print(f"Error: BERT model not found at {BERT_PATH}")
        exit()
    
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertModel.from_pretrained(BERT_PATH).to(DEVICE)
    
    # 2. 处理 FB15K
    fb_ents = read_entities(os.path.join(DATA_DIR, "FB15K_EntityTriples.txt"))
    generate_embeddings(fb_ents, os.path.join(DATA_DIR, "bert_attr_1.pkl"), tokenizer, model)
    
    # 3. 处理 DB15K
    db_ents = read_entities(os.path.join(DATA_DIR, "DB15K_EntityTriples.txt"))
    generate_embeddings(db_ents, os.path.join(DATA_DIR, "bert_attr_2.pkl"), tokenizer, model)
    
    print("\n✅ All text embeddings generated! You can now run main.py.")