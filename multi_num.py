import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from CNN import CNN

# 加载训练好的模型
model = torch.load("model/mnist_model.pkl", weights_only=False)
model = model.cuda()
model.eval()

# 预处理单个数字图像
def preprocess_single_image(image):
    image = cv2.resize(image, (28, 28))
    image = 255 - image
    image = image / 255.0
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).float()
    return image.cuda()

# 识别单个数字
def recognize_single_digit(image):
    input_image = preprocess_single_image(image)
    with torch.no_grad():
        output = model(input_image)
        _, pred = output.max(1)
        return pred.item()

# 数字定位与分割
def locate_and_segment_digits(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
    digit_boxes = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # 过滤掉小的噪声区域
            digit_image = gray[y:y + h, x:x + w]
            digit_images.append(digit_image)
            digit_boxes.append((x, y, w, h))
    # 按 x 坐标排序，确保数字按从左到右的顺序
    sorted_indices = np.argsort([box[0] for box in digit_boxes])
    sorted_digit_images = [digit_images[i] for i in sorted_indices]
    sorted_digit_boxes = [digit_boxes[i] for i in sorted_indices]
    return sorted_digit_images, sorted_digit_boxes

# 数字序列识别
def recognize_digit_sequence(image):
    digit_images, digit_boxes = locate_and_segment_digits(image)
    digit_sequence = []
    for digit_image in digit_images:
        digit = recognize_single_digit(digit_image)
        digit_sequence.append(digit)
    sequence_str = ''.join(map(str, digit_sequence))
    return sequence_str, digit_boxes

# 可视化展示
def visualize_recognition(image, digit_sequence, digit_boxes):
    for i, (x, y, w, h) in enumerate(digit_boxes):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(digit_sequence[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# 处理文件夹内的图片
def process_images_in_folder(folder_path, output_folder):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            digit_sequence, digit_boxes = recognize_digit_sequence(image)
            print(f"图片 {filename} 识别出的数字序列为: {digit_sequence}")
            result_image = visualize_recognition(image, digit_sequence, digit_boxes)
            cv2.imshow(f"Recognition Result - {filename}", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # 保存结果图片到输出文件夹
            result_path = os.path.join(output_folder, filename)
            cv2.imwrite(result_path, result_image)

if __name__ == "__main__":
    folder_path = "multi_num_images" 
    output_folder = "result_multi_num_images" 
    process_images_in_folder(folder_path, output_folder)