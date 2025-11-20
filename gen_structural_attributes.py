import os
import re
import h5py
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# ================= 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "FB15K-DB15K")
# 确保你已经下载了 models/bert-base-uncased 文件夹
BERT_PATH = os.path.join(SCRIPT_DIR, "models", "bert-base-uncased")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ================= 结构化属性名称读取函数 =================
def read_attribute_names(filename):
    print(f"Reading attribute names (predicates) from: {filename}")
    attribute_names = set()
    is_dbpedia = 'DB15K' in os.path.basename(filename)
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            attr_name = None
            if not is_dbpedia:  # FB15K style
                # Format: /m/06rf7 \t <http://rdf.freebase.com/ns/location.geocode.longitude> \t 9.70404945
                parts = line.split('\t')
                if len(parts) >= 2:
                    attr_name = parts[1].strip()
            else:  # DB15K style
                # Format: <entity> <predicate> "value"^^<datatype> .
                match = re.search(r'<[^>]+>\s+<(.*?)>', line)
                if match:
                    attr_name = match.group(1).strip()
            
            # 统一清理尖括号，确保与 kgs.py 内部键格式一致
            if attr_name and attr_name.startswith('<') and attr_name.endswith('>'):
                attr_name = attr_name[1:-1]
            
            if attr_name:
                attribute_names.add(attr_name)
    
    print(f"Found {len(attribute_names)} unique attribute names.")
    return list(attribute_names)


# ================= BERT 编码并保存 H5 文件 =================
def generate_embeddings_h5(attr_name_list, output_file, tokenizer, model):
    print(f"Generating embeddings for {output_file}...")
    
    batch_size = 32
    model.eval()
    
    # 1. 创建 H5 文件，准备存储
    with h5py.File(output_file, 'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        
        # 存储属性名称列表
        f.create_dataset('attr_name', data=np.array(attr_name_list, dtype=dt))
        
        # 准备存储嵌入的 dataset (BERT base dim = 768)
        attr_emb_dset = f.create_dataset('attr_emb', (len(attr_name_list), 768), dtype='float32')
        
        # 2. 编码并写入
        batches = [attr_name_list[i:i + batch_size] for i in range(0, len(attr_name_list), batch_size)]
        
        with torch.no_grad():
            current_index = 0
            for batch in tqdm(batches):
                
                texts = []
                for attr in batch:
                    # 策略：将 URI 或路径转换为可读文本 (例如：http://.../areaTotal -> area Total)
                    text = attr.replace('/', ' ').replace('_', ' ').replace('-', ' ').strip()
                    if 'http' in text:
                        match = re.search(r'/([^/]+)$', text)
                        text = match.group(1).replace('_', ' ') if match else text
                    texts.append(text)
                
                # 编码
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # 写入 H5 文件
                attr_emb_dset[current_index:current_index + len(embeddings)] = embeddings
                current_index += len(embeddings)
    
    print(f"\n✅ Saved {len(attr_name_list)} structural attribute embeddings to {output_file}")


# ================= 主流程 =================
if __name__ == "__main__":
    if not os.path.exists(os.path.join(BERT_PATH, "pytorch_model.bin")):
        print(
            f"Error: BERT model not found at {BERT_PATH}. Please ensure you have the 'models/bert-base-uncased' folder.")
        exit()
    
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertModel.from_pretrained(BERT_PATH).to(DEVICE)
    
    # 2. 处理 FB15K 结构化属性
    fb_attrs = read_attribute_names(os.path.join(DATA_DIR, "FB15K_NumericalTriples.txt"))
    generate_embeddings_h5(fb_attrs, os.path.join(DATA_DIR, "attr_name_1.h5"), tokenizer, model)
    
    # 3. 处理 DB15K 结构化属性
    db_attrs = read_attribute_names(os.path.join(DATA_DIR, "DB15K_NumericalTriples.txt"))
    generate_embeddings_h5(db_attrs, os.path.join(DATA_DIR, "attr_name_2.h5"), tokenizer, model)
    
    print("\n🎉 Structural attribute files (.h5) generated successfully! The training should now run.")