import torch
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            # 第一层卷积
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(2),

            # 第二层卷积
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(2),

            # 第三层卷积
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(2)
        )

        # 计算全连接层输入维度
        self.fc_input_size = self._calculate_fc_input_size()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_input_size, 128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _calculate_fc_input_size(self):
        # 模拟输入数据通过卷积层，计算输出的大小
        x = torch.randn(1, 1, 28, 28)
        x = self.conv(x)
        return x.view(1, -1).size(1)