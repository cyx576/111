import os
import json
import pickle
import h5py
import torch
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

# ================= 动态配置区 =================
CONFIG_FILE = "./args/tmea.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ===============================================

# 核心路径修复：使用脚本自身的绝对路径计算所有文件位置
# SCRIPT_DIR: /home/adl/Cyx/TMEA/TMEA-main/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR (文本数据目录): /home/adl/Cyx/TMEA/TMEA-main/data/FB15K-DB15K/
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "FB15K-DB15K")

# IMG_ROOT_DIR (图片数据目录): /home/adl/Cyx/TMEA/TMEA-main/data/image-graph_images
IMG_ROOT_DIR = os.path.join(SCRIPT_DIR, "data", "image-graph_images")

# 引入本地模型路径 (新)
LOCAL_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "vit-base-patch16-224-in21k")

print(f"Reading configuration from {CONFIG_FILE}...")

# 1. 加载配置（仅用于获取 dataset_division 等非路径参数）
try:
    with open(os.path.join(SCRIPT_DIR, CONFIG_FILE), 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_FILE}")
    exit()

print(f"Data Directory (Resolved): {DATA_DIR}")
print(f"Image Root Directory (Resolved): {IMG_ROOT_DIR}")
print(f"Local Model Directory: {LOCAL_MODEL_DIR}")  # 打印本地模型路径
print(f"Using Device: {DEVICE}")

if not os.path.exists(IMG_ROOT_DIR):
    print(f"Critical Error: Image folder not found at {IMG_ROOT_DIR}. Check the absolute path.")
    exit()
if not os.path.exists(DATA_DIR):
    print(f"Critical Error: Data folder not found at {DATA_DIR}. Check the absolute path.")
    exit()


# 辅助函数：安全保存 pickle 文件
def save_pickle(data, path):
    """Save data dictionary to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# 辅助函数：安全加载 pickle 文件 (支持断点续跑)
def load_pickle(path):
    """Load data dictionary from a pickle file, robustly handling errors."""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load pickle file {path}. Starting fresh. Error: {e}")
            return {}
    return {}


def load_ids(filename):
    """读取实体ID文件：将实体名映射到图片文件夹ID"""
    ent_to_id = {}
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found at {path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ent_to_id[parts[0]] = parts[1]
    return ent_to_id


def generate_image_features(ent_map, output_name):
    """
    生成图片特征 .pkl 文件，支持断点续跑和多图聚合。
    (现在假设 ent_map 中的 key 是干净的，不带尖括号)
    """
    out_path = os.path.join(DATA_DIR, output_name)
    print(f"\nProcessing images for {output_name}...")
    
    # --- 检查点加载逻辑 ---
    data_dict = load_pickle(out_path)
    
    all_entities = set(ent_map.keys())  # keys here are now CLEANED
    processed_entities = set(data_dict.keys())
    
    remaining_entities = all_entities - processed_entities
    
    if data_dict:
        print(f"--- RESUME MODE: Loaded {len(processed_entities)} processed entities. ---")
        print(f"--- Total entities: {len(all_entities)}. Remaining: {len(remaining_entities)} ---")
    else:
        print("--- STARTING NEW: No existing checkpoint found. ---")
    
    if not remaining_entities:
        print("All entities already processed. Skipping feature generation.")
        return
    
    # --- 模型加载逻辑 (保持不变) ---
    print(f"Initializing ViT model. Checking local path: {LOCAL_MODEL_DIR}")
    
    # 强制只尝试本地加载
    if os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")):
        print("ATTEMPTING LOCAL LOAD ONLY...")
        try:
            processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
            model = ViTModel.from_pretrained(LOCAL_MODEL_DIR).to(DEVICE)
            print("Successfully loaded model from local path.")
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"CRITICAL ERROR: Failed to load ViT model from local path: {LOCAL_MODEL_DIR}")
            print(f"Underlying Exception: {e}")
            print("The local model files are either damaged, a file is missing, or there is a library mismatch.")
            print(
                "Please ensure the 346MB pytorch_model.bin is fully intact and check file permissions again (chmod -R 755 .).")
            print("=" * 80 + "\n")
            raise
    else:
        print("ERROR: pytorch_model.bin not found in the local model directory. Halting.")
        return
    
    model.eval()  # 确保模型处于评估模式
    
    FEATURE_DIM = 768
    zero_vector = np.zeros(FEATURE_DIM).astype(np.float32)
    
    entity_list_to_process = list(remaining_entities)
    
    # --- 主处理循环 ---
    SAVE_FREQUENCY = 1000  # 每处理 1000 个实体保存一次进度
    
    pbar = tqdm(entity_list_to_process, desc=f"Extracting {output_name}")
    
    for i, ent_name in enumerate(pbar):  # ent_name 现在是 CLEANED key
        img_folder_id = ent_map[ent_name]
        entity_embeddings = []
        entity_folder = os.path.join(IMG_ROOT_DIR, img_folder_id)
        
        # 1. 查找文件夹内所有 .jpg 文件
        image_files = glob.glob(os.path.join(entity_folder, "*.jpg"))
        
        # 2. 处理所有找到的图片 (保持不变)
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                emb = outputs.pooler_output.cpu().numpy().flatten()
                entity_embeddings.append(emb)
            
            except Exception:
                continue
        
        # 3. 聚合特征
        if entity_embeddings:
            aggregated_emb = np.mean(entity_embeddings, axis=0)
            data_dict[ent_name] = [aggregated_emb]  # 使用 CLEANED 键保存
        else:
            data_dict[ent_name] = [zero_vector]  # 使用 CLEANED 键保存
        
        # 4. 检查点保存 (保持不变)
        if (i + 1) % SAVE_FREQUENCY == 0:
            save_pickle(data_dict, out_path)
            pbar.set_postfix_str(f"Checkpoint saved @ {len(data_dict)}")
    
    # 5. 最终保存
    save_pickle(data_dict, out_path)
    print(f"\nSuccessfully saved FINAL aggregated image features to {out_path}")


def generate_attr_placeholders(output_pkl, output_h5):
    """生成属性特征的占位符文件 (极简版)"""
    print(f"\nGenerating attribute placeholders for {output_pkl}...")
    
    # 1. 生成 .pkl (BERT 属性嵌入) 占位符
    out_path_pkl = os.path.join(DATA_DIR, output_pkl)
    if not os.path.exists(out_path_pkl):
        data_dict = {}
        save_pickle(data_dict, out_path_pkl)
        print(f"Saved {output_pkl} (Placeholder)")
    else:
        print(f"{output_pkl} already exists. Skipping.")
    
    # 2. 创建空的 .h5 文件 (attr_name) 占位符
    h5_path = os.path.join(DATA_DIR, output_h5)
    if not os.path.exists(h5_path):
        with h5py.File(h5_path, 'w') as f:
            pass
        print(f"Saved {output_h5} (Placeholder)")
    else:
        print(f"{output_h5} already exists. Skipping.")


# ================= 运行主逻辑 =================

if __name__ == "__main__":
    # 1. 加载映射文件 (ImageIndex.txt)
    fb_map = load_ids("FB15K_ImageIndex.txt")
    db_map = load_ids("DB15K_ImageIndex.txt")
    
    # 2. 生成图片特征 (clip_image_1.pkl 和 clip_image_2.pkl)
    generate_image_features(fb_map, "clip_image_1.pkl")
    generate_image_features(db_map, "clip_image_2.pkl")
    
    # 3. 生成属性特征占位符
    generate_attr_placeholders("bert_attr_1.pkl", "attr_name_1.h5")
    generate_attr_placeholders("bert_attr_2.pkl", "attr_name_2.h5")
    
    print("\n\nAll preprocessing steps for TMEA initialization are complete. You can now run python main.py")