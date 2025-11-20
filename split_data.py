import os
import random
import json

# ================ 配置区 (读取 JSON) ================
CONFIG_FILE = "./args/tmea.json"
# ===================================================

print(f"Reading configuration from {CONFIG_FILE}...")
# 1. 加载配置
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# 2. 从配置中获取动态参数
root_path = config["training_data"]
division_path = config["dataset_division"] # <--- 这里读取了新的可读路径

source_file = "DB15K_SameAsLink.txt" # 原始对齐文件
output_dir = os.path.join(root_path, division_path) # 动态构建输出目录

# 创建目录结构
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created dynamic directory: {output_dir}")

# 读取所有对齐数据
links = []
full_source_path = os.path.join(root_path, source_file)
if not os.path.exists(full_source_path):
    print(f"Error: Source file not found at {full_source_path}")
    exit()

with open(full_source_path, 'r', encoding='utf-8') as f:
    for line in f:
        links.append(line.strip())

# 打乱顺序
random.seed(2025)
random.shuffle(links)

# 计算划分数量 (20% 训练, 10% 验证, 70% 测试)
total = len(links)
n_train = int(total * 0.2)
n_valid = int(total * 0.1)
n_test = total - n_train - n_valid

train_links = links[:n_train]
valid_links = links[n_train:n_train+n_valid]
test_links = links[n_train+n_valid:]

print(f"Total links: {total}")
print(f"Train links: {len(train_links)} | Valid links: {len(valid_links)} | Test links: {len(test_links)}")


# 将分割结果写入文件，并格式化为 ent1 ent2
def format_and_write_links(links_list, filename, output_dir):
    """
    将链接列表写入文件，格式化为: entity1 \t entity2
    """
    full_path = os.path.join(output_dir, filename)
    formatted_count = 0
    
    with open(full_path, 'w', encoding='utf-8') as f:
        for link_line in links_list:
            # 原始行: /m/01m4kpp <SameAs> <http://dbpedia.org/resource/Andy_Griffith> .
            params = link_line.strip().split()
            
            # 检查格式：如果不是 4 个元素，跳过
            if len(params) == 4 and params[1] == '<SameAs>':
                ent1 = params[0]  # /m/01m4kpp
                ent2 = params[2]  # <http://dbpedia.org/resource/Andy_Griffith>
                
                # 关键修复：使用 Tab 键 ('\t') 连接两个实体
                f.write(f"{ent1}\t{ent2}\n")
                formatted_count += 1
            # 否则，可能是空行或其他格式错误，跳过
            elif len(link_line.strip()) > 0:
                print(f"Warning: Skipping improperly formatted link line: {link_line}")
    
    print(f"Saved {formatted_count} links (formatted) to {full_path}")


# 写入文件
format_and_write_links(train_links, 'train_links', output_dir)
format_and_write_links(valid_links, 'valid_links', output_dir)
format_and_write_links(test_links, 'test_links', output_dir)

print("\nLink splitting and formatting complete.")
