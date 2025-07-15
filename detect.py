import torch
import os
import pandas as pd
from model import AlexNet
from util import process_video_data,load_data
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="推理脚本参数")

    parser.add_argument('--frames_len', type=int, default=16,
                        help='堆叠的视频帧数，默认 16')

    parser.add_argument('--timestamp', type=str,
                        default=None,
                        help='日志/模型目录时间戳，比如 2025-07-15_11-06-14（默认使用当前时间）')

    return parser.parse_args()


def main():
    #默认参数
    args = parse_args()
    # 使用命令行参数
    frames_len = args.frames_len
    # 如果没有传 timestamp，就用当前时间
    timestamp = args.timestamp 

    #路径设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root="./result"
    path=os.path.join(root,timestamp)
    model_dir="./models"
    model_path=os.path.join(path,model_dir,"AlexNet.pt")
    labels_file = './save/labels.txt'
    video_dir = '../补充版/正式数据集' 
    output_csv = os.path.join(path,'inference_results.csv')
 
    # ===== 加载标签 =====
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    # ===== 加载模型 =====
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # ===== 收集所有待推理的视频路径 =====
    video_folders =[]
    for name in os.listdir(video_dir):
        video_folders.append(os.path.join(video_dir, name))
    print(video_folders)

    # ===== 推理并保存结果 =====
    results = []


    for video_path in video_folders:
        video_name = os.path.basename(video_path)
        try:
            # 加载视频帧数据
            video_tensor = process_video_data([video_path], frames_len=frames_len)[0]  # 取出第一个视频张量
            video_tensor = video_tensor.unsqueeze(0) 
            video_tensor = video_tensor.to(device)

            # 从路径中提取真实标签（如果能）
            true_label_str = video_name.split('_')[2]  # 例如 '005'
            true_label_index = labels.index(true_label_str) if true_label_str in labels else -1

            # 推理
            with torch.no_grad():
                output = model(video_tensor)
                pred_idx = output.argmax(1).item()
                pred_label = labels[pred_idx]

            print(f"{video_name}: Pred = {pred_label}, True = {true_label_str}")

            results.append({
                'video_name': video_name,
                'true_label_index': true_label_index,
                'true_label_str': true_label_str,
                'prediction_index': pred_idx,
                'prediction_label': pred_label
            })

        except Exception as e:
            print(f"❌ 推理失败: {video_name} - {str(e)}")


    # ===== 保存为 CSV =====
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ 推理结果已保存到 {output_csv}")

if __name__ == "__main__":
    main()
