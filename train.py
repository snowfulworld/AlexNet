import os
import csv
import time
from util import read_from_csv,process_video_data,load_data,init_weight,plot_metrics
import torch
from model import AlexNet
from torch import nn
from datetime import datetime
import pandas as pd
import onnx

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sign Language Recognition Training")

    parser.add_argument('--frames_len', type=int, default=16, help='视频堆叠帧数')
    parser.add_argument('--batch_size', type=int, default=16, help='每批样本数')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')

    return parser.parse_args()


def main():
    root="./result"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path=os.path.join(root,timestamp)
    os.makedirs(path, exist_ok=True)
    profile_path='./save' 
    model_dir="models"
    model_path=os.path.join(path,model_dir)
    log_dir="logs"
    log_path=os.path.join(path,log_dir)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    best_model_path = os.path.join(model_path,'AlexNet.pt')
    best_onnx_path=os.path.join(model_path,'AlexNet.onnx')
    # 默认参数设置
    args = parse_args()

    frames_len = args.frames_len
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    #读取数据
    labels_file=os.path.join(profile_path,'labels.txt')
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    
    # 读取 train.csv 和 test.csv 文件
    train_file=os.path.join(profile_path,'train.csv')
    test_file=os.path.join(profile_path,'test.csv')
    train_data_path, train_labels = read_from_csv(train_file)
    test_data_path, test_labels = read_from_csv(test_file)
    #将视频帧进行堆叠
    train_data = process_video_data(train_data_path, frames_len=frames_len)
    test_data = process_video_data(test_data_path, frames_len=frames_len)
    #将标签变成数字类型
    train_labels = [labels.index(item) for item in train_labels if item in labels]
    test_labels = [labels.index(item) for item in test_labels if item in labels]
    
    train_iter=load_data(train_data,train_labels,batch_size)
    test_iter=load_data(test_data,test_labels,batch_size)
    for x,y in train_iter:
        print(x.shape)
        break
    
    
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备为"+str(device))
    #模型
    model=AlexNet(input_frames=frames_len,num_classes=len(labels)).to(device)
    model.apply(init_weight)
    #损失函数和优化器
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)#0.001会导致loss为nan
    # 初始化最小测试损失
    best_test_loss = float('inf')
    #训练
    train_len=len(train_iter.dataset)
    test_len = len(test_iter.dataset)
    print(test_len)
    all_train_acc, all_train_loss = [], []
    all_test_acc, all_test_loss = [], []

    best_test_loss = float('inf')
    best_epoch = 0
    shape = None

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            hat_y = model(x)
            loss = loss_fn(hat_y, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_acc += (hat_y.argmax(1) == y).sum().item()

        train_acc = epoch_acc / train_len
        all_train_acc.append(train_acc)
        all_train_loss.append(epoch_loss)

        # === 测试 ===
        model.eval()
        test_acc = 0
        test_loss = 0
        with torch.no_grad():
            for x, y in test_iter:
                x, y = x.to(device), y.to(device)
                shape = x.shape
                hat_y = model(x)
                test_loss += loss_fn(hat_y, y).item()
                test_acc += (hat_y.argmax(1) == y).sum().item()

        test_acc /= test_len
        all_test_acc.append(test_acc)
        all_test_loss.append(test_loss)

        print(f"[{epoch+1}/{epochs}] ✅ Test Acc: {test_acc:.4f} | Loss: {test_loss:.4f}")

        # === 保存最优模型 ===
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            torch.save(model,best_model_path)
            dummy_input = torch.randn(shape).to(device)
            torch.onnx.export(model, dummy_input, best_onnx_path, opset_version=11)
            print(f"🚀 Best model saved at epoch {best_epoch} with test loss {best_test_loss:.4f}")

        # === 每轮保存图（覆盖） ===
        plot_metrics(
            all_train_loss, all_test_loss,
            all_train_acc, all_test_acc,
            save_path=os.path.join(log_path, "current.png"),
            annotate_last=False,
            epoch=epoch+1
        )
    end_time = time.time()
    total_time = end_time - start_time

    # === 保存 CSV ===
    df = pd.DataFrame({
        'epoch': list(range(1, epochs+1)),
        'train_acc': all_train_acc,
        'train_loss': all_train_loss,
        'test_acc': all_test_acc,
        'test_loss': all_test_loss
    })
    df.to_csv(os.path.join(log_path, "training_log.csv"), index=False)

    # === 写入 TXT 总结 ===
    with open(os.path.join(log_path, "results.txt"), "w") as f:
        f.write("训练结果总结\n")
        f.write("====================\n")
        # 🔧 参数记录部分
        f.write(f"参数设置：\n")
        f.write(f"  - 帧数（frames_len）: {frames_len}\n")
        f.write(f"  - 批大小（batch_size）: {batch_size}\n")
        f.write(f"  - 学习率（learning_rate）: {learning_rate}\n")
        f.write(f"  - 总训练轮数（epochs）: {epochs}\n\n")

        # ✅ 最终训练结果
        f.write(f"最后一轮训练准确率（Train Acc）: {all_train_acc[-1]:.4f}\n")
        f.write(f"最后一轮训练损失（Train Loss）: {all_train_loss[-1]:.4f}\n")
        f.write(f"最后一轮测试准确率（Test Acc）: {all_test_acc[-1]:.4f}\n")
        f.write(f"最后一轮测试损失（Test Loss）: {all_test_loss[-1]:.4f}\n\n")

        # ✅ 最优结果
        f.write(f"最优模型出现在第 {best_epoch} 轮\n")
        f.write(f"最优测试损失（Best Test Loss）: {best_test_loss:.4f}\n")
        f.write(f"⏱️ 总训练时间: {total_time:.2f} 秒\n")
        f.write("====================\n")

    print(f"\n🎉 训练完成！结果保存于：{log_path}")


if __name__ == "__main__":
    main()