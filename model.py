import torch.nn as nn
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class AlexNet(nn.Module):
    def __init__(self, input_frames=64, num_classes=10):
        """
        手语识别3D CNN模型 (基于2D CNN处理视频帧)
        
        参数:
            input_frames (int): 输入视频帧数 (默认64帧)
            num_classes (int): 分类类别数 
        """
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # 输入: (batch_size, frames_len, 224, 224)
            nn.Conv2d(input_frames, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 使用示例 (在其他文件中)
if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")  # 应显示 nightly 版本
    print(torch.cuda.get_arch_list())
    print(f"GPU可用: {torch.cuda.is_available()}")  # 必须返回 True
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")  # 应显示 RTX 5070
    print(f"计算能力: {torch.cuda.get_device_capability()}")  # 应显示 (12,0)
    # 1. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AlexNet(input_frames=64, num_classes=10).to(device)
    
    # 2. 打印模型结构
    print(model)
    
    # 3. 测试输入输出
    dummy_input = torch.randn(2, 64, 224, 224).to(device) 
    output = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")