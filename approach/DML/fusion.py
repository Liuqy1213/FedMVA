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


# 将数据集按target字段分组的函数
def group_by_target(data):
    groups = {0: [], 1: [], 2: []}
    for entry in data:
        target = entry["target"][0][0]  # 假设 target 的格式为 [[n]]
        if target in groups:
            groups[target].append(entry)
    return groups


# 主函数，处理并融合特征
def main(cpg_file, code_file, output_file):
    # 读取CPG和CODE数据
    cpg_data = read_json(cpg_file)
    code_data = read_json(code_file)

    # 分组数据
    cpg_groups = group_by_target(cpg_data)
    code_groups = group_by_target(code_data)

    output_data = []
    unmatched_cpg_count = 0
    unmatched_code_count = 0

    for target in [0, 1, 2]:
        # 获取相同target的CPG和CODE条目组
        cpg_entries = cpg_groups.get(target, [])
        code_entries = code_groups.get(target, [])

        # 计算每个target组中能匹配的最小条目数
        min_len = min(len(cpg_entries), len(code_entries))

        # 进行匹配和特征融合
        for i in range(min_len):
            fused_feature = np.concatenate((cpg_entries[i]["graph_feature"], code_entries[i]["graph_feature"]))
            output_entry = {
                "target": [[target]],
                "graph_feature": fused_feature.tolist()
            }
            output_data.append(output_entry)

        # 记录未匹配的条目数量
        unmatched_cpg_count += len(cpg_entries) - min_len
        unmatched_code_count += len(code_entries) - min_len

    # 输出匹配信息
    print(f"Number of matched entries: {len(output_data)}")
    print(f"Number of unmatched CPG entries: {unmatched_cpg_count}")
    print(f"Number of unmatched CODE entries: {unmatched_code_count}")

    # 写入输出到新的JSON文件
    if output_data:
        write_json(output_data, output_file)
        print(f"Fusion completed. Output saved to {output_file}")
    else:
        print("Warning: No data was written to the output file. Check file name matching and data integrity.")


# 示例用法
cpg_file = 'D:\\VDBFL\\graph_feature_extraction\\SDV_After_GNN\\datasets-line-ggnn.json'  # CPG特征文件路径
code_file = 'D:\\VDBFL\\data\\parted_embeddings_reduced.json'  # CODE特征文件路径
output_file = 'D:\\VDBFL\\graph_feature_extraction\\SDV_After_GNN\\datasets-fused-ggnn.json'  # 输出文件路径

main(cpg_file, code_file, output_file)
