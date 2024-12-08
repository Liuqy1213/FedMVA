import os
import json
import torch
import numpy as np
from transformers import RobertaTokenizer, T5Model
import re
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

# 加载 CodeT5 的分词器和模型
tokenizer = RobertaTokenizer.from_pretrained('D:\\VDBFL\\data_processing\\codet5-base')
model = T5Model.from_pretrained('D:\\VDBFL\\data_processing\\codet5-base')
model.eval()

# 嵌入生成函数
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    return embeddings

# 设置输入目录和输出文件
input_dir = "D:\VDBFL\data\conjoint_code_comments"  # 处理合并后的 JSON 文件的目录
output_file_path = "D:\\VDBFL\\data\\conjoint_embeddings.json"

# 初始化一个列表来存储所有嵌入数据
all_embeddings = []

# 获取输入目录中的所有文件，并过滤出需要处理的文件
json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# 遍历所有文件，处理 code 和 comment 嵌入，显示进度条
for file_name in tqdm(json_files, desc="Processing files", unit="file"):
    file_path = os.path.join(input_dir, file_name)

    # 尝试读取合并后的 JSON 文件
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            code_data = data.get("code_tokens", [])
            comment_data = data.get("comments", [])
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue  # 如果出错，跳过该文件

    # 处理代码嵌入
    code_embeddings = torch.zeros((1, 768))  # 默认初始化为零向量
    if code_data:
        code_text = " ".join(code_data)
        try:
            code_embeddings = generate_embeddings(code_text)
        except Exception as e:
            print(f"Error processing code: {e}")

    # 处理注释嵌入
    comment_embeddings = torch.zeros((1, 768))  # 默认初始化为零向量
    if comment_data:
        comment_text = " ".join(comment_data)
        try:
            comment_embeddings = generate_embeddings(comment_text)
        except Exception as e:
            print(f"Error processing comment: {e}")

    # 提取标签，假设标签是文件名中的数字部分
    label_match = re.search(r'_(\d+)', file_name)  # 假设标签在文件名中
    label = int(label_match.group(1)) if label_match else 0  # 默认设置为0，或根据需要处理

    output_data = {
        "target": [[label]],  # 设置为与 CPG 一致的标签
        "graph_feature": np.concatenate((code_embeddings.numpy(), comment_embeddings.numpy()), axis=1).tolist()
    }

    # 将每个嵌入结果添加到总列表中
    all_embeddings.append(output_data)

# 最后，将所有嵌入写入一个 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(all_embeddings, output_file, indent=4)
    print(f"Saved all embeddings to {output_file_path}")

print("Processing completed for all code and comment files.")
