import os
import json
import argparse
from tqdm import tqdm


def raw_code2dict(file_path):
    file_name = os.path.basename(file_path)
    if file_name.split('.')[-1] == 'c':
        try:
            label = int(file_name[-3])  # 提取标签
        except ValueError:
            raise ValueError(f"Unexpected file name format for {file_name}")

        try:
            code = open(file_path, 'r', encoding='utf-8').read()
        except UnicodeDecodeError:
            print(f"Failed to decode {file_path}")
            return None

        output = {
            'file_name': file_name,
            'label': label,
            'code': code
        }
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='name of project for differentiating files')
    parser.add_argument('--input', help='directory where raw code and parsed are stored',
                        default='D:\\VDBFL\\new_dataset\\')
    parser.add_argument('--output', help='output directory for resulting json file',
                        default='D:\\VDBFL\\new_dataset\\ggnn_input')
    args = parser.parse_args()

    code_file_path = args.input
    output_data = []
    for cfile in tqdm(os.listdir(code_file_path)):
        fp = os.path.join(code_file_path, cfile)
        data_entry = raw_code2dict(fp)
        if data_entry is not None:
            output_data.append(data_entry)

    output_file = os.path.join(args.output, 'bigvul_cpg_full_text_files.json')
    with open(output_file, 'w') as of:
        json.dump(output_data, of)

    print(f'Saved Output File to {output_file}')
