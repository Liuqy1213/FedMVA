import json
import numpy as np


# 读取JSON文件的函数
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 保存输出到JSON文件的函数
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# 处理单条数据的节点特征，计算平均值并扩展维度
def process_single_entry(entry):
    # 检查是否包含 'node_features' 键
    if 'node_features' in entry:
        node_features = np.array(entry['node_features'])  # 将节点特征转换为numpy数组
    else:
        raise KeyError("The key 'node_features' was not found in the entry.")

    # 计算所有节点特征的平均值
    avg_feature = np.mean(node_features, axis=0)

    # # 扩展或截取平均特征到200维
    # if avg_feature.shape[0] < 200:
    #     extended_feature = np.pad(avg_feature, (0, 200 - avg_feature.shape[0]), mode='constant')
    # else:
    #     extended_feature = avg_feature[:200]
    #
    # return extended_feature
    return avg_feature


# 主函数，处理输入输出
def main(input_file, output_file):
    # 读取输入数据
    data = read_json(input_file)

    output_data = []

    # 遍历数据中的每一条，处理特征并保存
    for entry in data:
        avg_feature = process_single_entry(entry)
        output_entry = {
            'target': entry['targets'],  # 获取每个条目的'targets'
            'graph_feature': avg_feature.tolist()  # 将numpy数组转换为列表以便保存为JSON格式
        }
        output_data.append(output_entry)

    # 写入输出到新的JSON文件
    write_json(output_data, output_file)


# 示例用法
input_file = 'D:\\VDBFL\\data_processing\\prep_script\\bigvul-full_graph.json'
output_file = 'D:\\VDBFL\\graph_feature_extraction\\SDV_After_GNN\\datasets-line-ggnn.json'

main(input_file, output_file)
