import cv2
import torch
import torchvision.transforms as transforms
import os
from CNN import CNN

# 预处理图像
def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 调整图像大小为28x28
    image = cv2.resize(image, (28, 28))
    # 反转颜色（因为MNIST数据集是白底黑字）
    image = 255 - image
    # 二值化处理
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # 归一化像素值到0-1之间
    image = image / 255.0
    # 将图像转换为张量
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)  # 添加一个维度以匹配模型输入
    # 将输入数据转换为float32类型
    image = image.float()
    return image

# 加载训练好的模型
model = torch.load("model/mnist_model.pkl", weights_only=False)
model = model.cuda()
model.eval()

# 进行预测
def predict_digit(image_path):
    image = preprocess_image(image_path)
    if image is None:
        return None
    image = image.cuda()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# 定义存放图片的文件夹路径
image_folder = r"images" 

# 获取文件夹内所有图片的路径
image_extensions = ['.jpg', '.jpeg', '.png']
image_paths = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(root, file))

# 遍历图像路径列表进行预测
for i, image_path in enumerate(image_paths):
    predicted_digit = predict_digit(image_path)
    if predicted_digit is not None:
        print(f"第 {i + 1} 张图像 ({image_path}) 预测的数字是: {predicted_digit}")