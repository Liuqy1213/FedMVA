import clang.cindex
import json
import os

# 初始化 clang.cindex，设置库路径和库文件（根据安装位置修改）
clang.cindex.Config.set_library_path("D:/Users/admin/anaconda3/envs/pytorch/Lib/site-packages/LLVM/lib")
clang.cindex.Config.set_library_file("D:/Users/admin/anaconda3/envs/pytorch/Lib/site-packages/LLVM/bin/libclang.dll")

# 定义提取函数
class CodeExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.index = clang.cindex.Index.create()
        self.comments = []
        self.code_tokens = []

    def extract(self):
        # 解析文件
        translation_unit = self.index.parse(self.file_path)
        # 遍历 AST，提取词法信息和注释信息
        self.traverse_ast(translation_unit.cursor)

    def traverse_ast(self, cursor):
        # 提取节点中的代码词法信息和注释
        for token in cursor.get_tokens():
            if token.kind == clang.cindex.TokenKind.COMMENT:
                self.comments.append(token.spelling)  # 提取注释
            else:
                self.code_tokens.append(token.spelling)  # 提取代码词法内容

        # 递归处理子节点
        for child in cursor.get_children():
            self.traverse_ast(child)

    def save_to_file(self, output_path):
        # 将词法数据和注释信息保存到一个文件中
        output_data = {
            "code_tokens": self.code_tokens,
            "comments": self.comments
        }
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, indent=4)

# 主函数
def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 BigVul 数据集中的文件
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_name.endswith(".c") or file_name.endswith(".cpp"):  # 只处理 C/C++ 文件
            print(f"Processing file: {file_name}")

            # 初始化 CodeExtractor，并提取代码和注释信息
            extractor = CodeExtractor(file_path)
            extractor.extract()

            # 输出路径
            output_path = os.path.join(output_dir, f"{file_name}.json")

            # 保存数据
            extractor.save_to_file(output_path)
            print(f"Data saved for {file_name}")

# 设置数据集输入和输出目录
input_dir = "D:\\VDBFL\\new_dataset"  # 替换为 BigVul 数据集路径
output_dir = "D:\\VDBFL\\data\\conjoint_code_comments"  # 替换为保存提取数据的输出路径
main(input_dir, output_dir)