import os
import random
import argparse
from collections import defaultdict
import csv

def main():
    # 默认参数设置
    default_root = '../补充版/正式数据集' 
    default_save_dir = './save'
    default_test_ratio = 0.2
    default_random_seed = 0

    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('-i', '--input', type=str, default=default_root, 
                        help=f'输入数据集目录 (默认: {default_root})')
    parser.add_argument('-o', '--output', type=str, default=default_save_dir,
                        help=f'输出保存目录 (默认: {default_save_dir})')
    parser.add_argument('-r', '--ratio', type=float, default=default_test_ratio,
                        help=f'测试集比例 (默认: {default_test_ratio})')
    parser.add_argument('-s', '--seed', type=int, default=default_random_seed,
                        help=f'随机种子 (默认: {default_random_seed})')
    args = parser.parse_args()

    # 2. 使用参数（命令行参数优先于代码默认值）
    root = args.input
    save_dir = args.output
    test_ratio = args.ratio
    random_seed = args.seed

    # 3. 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 4. 处理数据集
    video_paths = os.listdir(root)
    labels = []
    videos_path = []
    
    for video_path in video_paths:
        if video_path.lower().endswith('.mp4'):
            full_path = os.path.join(root, video_path)
            videos_path.append(full_path)
            label = video_path.split("_")[2]
            if label not in labels:
                labels.append(label)

    # 保存标签
    labels_file = os.path.join(save_dir, 'labels.txt')
    with open(labels_file, 'w', encoding='utf-8') as f:
        for label in sorted(labels):
            f.write(f"{label}\n")

    # 数据集划分函数
    def train_test_split_by_label(data_path, ratio=test_ratio, seed=random_seed):
        train_data_path = []; train_labels = []
        test_data_path = []; test_labels = []
        label_groups = defaultdict(list)

        for path in data_path:
            label = path.split("_")[2]
            label_groups[label].append(path)

        random.seed(seed)

        for label, group in label_groups.items():
            if len(group) == 0:
                continue
            
            n_test = max(1, int(len(group) * ratio))
            test_samples = random.sample(group, n_test)
            
            test_data_path.extend(test_samples)
            test_labels.extend([label] * n_test)
            
            train_samples = [x for x in group if x not in test_samples]
            train_data_path.extend(train_samples)
            train_labels.extend([label] * len(train_samples))

        return train_data_path, train_labels, test_data_path, test_labels

    # 执行划分
    train_data, train_lbls, test_data, test_lbls = train_test_split_by_label(videos_path)

    # 保存结果
    def save_to_csv(filename, data, labels):
        with open(os.path.join(save_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'label'])
            writer.writerows(zip(data, labels))

    save_to_csv('train.csv', train_data, train_lbls)
    save_to_csv('test.csv', test_data, test_lbls)

    # 打印摘要
    print("\n数据处理完成！")
    print(f"输入目录: {os.path.abspath(root)}")
    print(f"输出目录: {os.path.abspath(save_dir)}")
    print(f"总标签数: {len(labels)}")
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    print(f"测试集比例: {test_ratio:.1%}")
    print("\n生成文件:")
    print(f"  - {os.path.join(save_dir, 'labels.txt')}")
    print(f"  - {os.path.join(save_dir, 'train.csv')}")
    print(f"  - {os.path.join(save_dir, 'test.csv')}")

if __name__ == "__main__":
    main()