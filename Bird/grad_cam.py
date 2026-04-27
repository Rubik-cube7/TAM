import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from inference import load_model 
from numpy.lib.stride_tricks import sliding_window_view

def apply_rank_gaussian_filter(cam_map, k=3):
    """
    Rank Gaussian Filter (秩高斯滤波器)
    用于平滑 Grad-CAM 激活图
    """
    H, W = cam_map.shape
    pad = k // 2
    padded = np.pad(cam_map, pad, mode='reflect')
    windows = sliding_window_view(padded, (k, k)).reshape(H, W, -1)
    sorted_windows = np.sort(windows, axis=-1)
    
    mu = np.mean(windows, axis=-1, keepdims=True)
    sigma = np.std(windows, axis=-1, keepdims=True)
    
    eps = 1e-6
    mu = np.where(mu < eps, eps, mu)
    cv = sigma / mu
    cv = np.where(cv < eps, eps, cv)
    
    m = (k * k) // 2 
    ranks = np.arange(k * k).reshape(1, 1, -1) 
    
    exponent = - ((ranks - m) ** 2) / (2 * (cv ** 2))
    weights = np.exp(exponent)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    
    filtered_cam = np.sum(sorted_windows * weights, axis=-1)
    filtered_cam = (filtered_cam - np.min(filtered_cam)) / (np.max(filtered_cam) - np.min(filtered_cam) + 1e-8)
    
    return filtered_cam

def generate_cam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('model.pth', device)

    # 【核心修改区 1】：EfficientNet 的最后一层卷积层是 features[-1]
    target_layers = [model.features[-1]] 
    
    cam = GradCAM(model=model, target_layers=target_layers)

    # 【核心修改区 2】：适配 B4 的高分辨率
    preprocess = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    samples_dir = './samples'
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)
    
    total_images = 0
    correct_predictions = 0
    
    print("开始基于 EfficientNet-B4 生成 Rank-Gaussian 优化后的热力图...")
    
    for true_class_idx, class_folder in enumerate(sorted(os.listdir(samples_dir))):
        class_path = os.path.join(samples_dir, class_folder)
        if not os.path.isdir(class_path): continue
        
        save_class_dir = os.path.join(result_dir, class_folder)
        os.makedirs(save_class_dir, exist_ok=True)
            
        for img_name in os.listdir(class_path):
            if img_name.startswith('cam_'): continue
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                
            img_path = os.path.join(class_path, img_name)
            
            # 【核心修改区 3】：底图也放大到 380，生成更高清的热力图
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1] 
            rgb_img_resized = cv2.resize(rgb_img, (380, 380))
            rgb_img_float = np.float32(rgb_img_resized) / 255
            
            from PIL import Image
            pil_img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                pred_class = predicted_idx.item()
            
            total_images += 1
            if pred_class == true_class_idx:
                correct_predictions += 1
                status = "✅ 正确"
            else:
                status = f"❌ 错误 (错认类别 {pred_class})"

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # 使用秩高斯滤波降噪
            grayscale_cam_filtered = apply_rank_gaussian_filter(grayscale_cam, k=3)
            
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam_filtered, use_rgb=True)
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
            save_name = f"cam_{img_name}"
            save_path = os.path.join(save_class_dir, save_name)
            cv2.imwrite(save_path, visualization_bgr)
            
            print(f"处理完成: {class_folder}/{img_name} | {status}")

    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print("\n" + "="*50)
        print(" 优化版 Grad-CAM 热力图生成全部完成！")
        print(f" Sample集准确率: {accuracy:.2f}%")
        print("="*50 + "\n")

if __name__ == '__main__':
    generate_cam()