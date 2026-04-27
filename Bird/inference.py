import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

NUM_CLASSES = 200

def load_model(model_path='model.pth'): 
    """ 
    加载训练好的模型
    Args:
        model_path (str): 模型权重文件路径，默认为 'model.pth' 
    Returns:
        torch.nn.Module: 加载好权重的模型
    """ 
    # 自动识别设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化 EfficientNet-B4
    model = models.efficientnet_b4(weights=None)
    
    # 替换分类头
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model

def predict(model, image_path): 
    """ 
    对单张图片进行预测
    Args:
        model (torch.nn.Module): load_model()返回的模型
        image_path (str): 图片的完整路径
    Returns:
        int: 预测的类别编号（0-199）
    """ 
    # 自动获取模型所在设备
    device = next(model.parameters()).device
    
    # 严格的预处理操作
    preprocess = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图片并转为RGB
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 推理并返回结果
    model.eval() 
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    return int(predicted_idx.item())