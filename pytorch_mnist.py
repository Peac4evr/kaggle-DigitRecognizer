import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

# 数据加载
train_data = datasets.MNIST(
    root="mnist",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="mnist",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# 分批加载数据
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

# 初始化模型
cnn = CNN()
cnn = cnn.cuda()

# 损失函数
loss_func = torch.nn.CrossEntropyLoss()

# 优化函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练过程
num_epochs = 20
for epoch in range(num_epochs):
    for index, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # 前向传播
        outputs = cnn(images)

        # 计算损失函数
        loss = loss_func(outputs, labels)

        # 清空梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        print("当前为第{}轮，当前批次为{}/{},loss为{}".format(epoch + 1, index + 1, len(train_data) // 64 + 1, loss.item()))

    # 学习率调度
    scheduler.step()

    # 测试集验证
    loss_test = 0
    rightValue = 0
    for index2, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        loss_test += loss_func(outputs, labels)

        _, pred = outputs.max(1)
        rightValue += (pred == labels).sum().item()

        print("当前为第{}轮测试集验证，当前批次为{}/{},loss为{}，准确率是{}".format(epoch + 1, index2 + 1, len(test_data) // 64 + 1, loss_test,
                                                                   rightValue / len(test_data)))

torch.save(cnn, "model/mnist_model.pkl")