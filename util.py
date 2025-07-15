import csv
import torch
import os
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取 CSV 文件的函数
def read_from_csv(filename):
    data = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            data.append(row[0])  
            labels.append(row[1]) 
    return data, labels


def stack_frames(frames, frames_len):
    """
    将一系列帧堆叠成一个张量。
    """
    return torch.cat(frames[:frames_len], dim=0)  # 根据 frames_len 保证堆叠的帧数

# 函数：读取视频路径并堆叠帧
def process_video_data(video_data_path, frames_len=4):
    """
    处理视频数据，将每个视频的帧堆叠成一个张量，并进行标准化。
    输入:
        video_data_path: 视频文件夹路径列表
        frames_len: 堆叠的帧数，默认为 4
    输出:
        data: 堆叠后的图像数据列表
    """
    data = []

    for video in video_data_path:
        print(f"Processing video folder: {video}")  # 打印视频文件夹路径
        
        frames = []
        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Error: Could not open video {video}.")
            continue

        # 读取视频的每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 读取完毕
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2GRAY)  # 转灰度图
            frame = torch.tensor(frame, dtype=torch.float32)  # 转换为张量
            frame = frame.unsqueeze(0)  # 添加一个维度，变为 [1, 224, 224]
            
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
            frame = normalize(frame)
            
            frames.append(frame)
            frame_count += 1
        
        cap.release()  # 释放视频文件

        # 打印当前视频的帧数
        print(f"Frames count for video {video}: {frame_count}")

        # 如果读取的帧数不足，跳过
        if frame_count < frames_len:
            print(f"Warning: Video {video} has fewer frames than required ({frames_len}). Skipping.")
            continue

        # 使用 stack_frames 函数堆叠帧
        data.append(stack_frames(frames, frames_len))
    
    return data
#加载数据
class MyDatasets(Dataset):
    def __init__(self,data,labels,size=None):
        self.data=data
        self.labels=labels
        self.size=size
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img=self.data[index]
        label=torch.tensor(self.labels[index],dtype=torch.long)
        return img,label
    
def load_data(data,labels,batch_size,size=None):
    result=MyDatasets(data,labels)
    return DataLoader(result,batch_size,shuffle=True,num_workers=8)
#初始化权重
def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
#可视化
def plot_metrics(train_loss, test_loss, train_acc, test_acc, save_path, annotate_last=False, epoch=None):
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve' + (f' (Epoch {epoch})' if epoch else ''))
    plt.legend()
    if annotate_last:
        plt.text(len(train_loss)-1, train_loss[-1], f'{train_loss[-1]:.4f}', ha='right', va='bottom')
        plt.text(len(test_loss)-1, test_loss[-1], f'{test_loss[-1]:.4f}', ha='right', va='bottom')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve' + (f' (Epoch {epoch})' if epoch else ''))
    plt.legend()
    if annotate_last:
        plt.text(len(train_acc)-1, train_acc[-1], f'{train_acc[-1]:.4f}', ha='right', va='bottom')
        plt.text(len(test_acc)-1, test_acc[-1], f'{test_acc[-1]:.4f}', ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
