import os
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_utils import process_vision_info

# ==========================================
# 1. 定义 Hook 类，用于捕获前向特征与反向梯度
# ==========================================
class HookBase:
    def __init__(self, module):
        self.hook_f = module.register_forward_hook(self.hook_fn_f)
        self.hook_b = module.register_full_backward_hook(self.hook_fn_b)
        self.features = None
        self.gradients = None

    def hook_fn_f(self, module, input, output):
        # 取出前向传播的特征
        if isinstance(output, tuple):
            self.features = output[0]
        else:
            self.features = output

    def hook_fn_b(self, module, grad_in, grad_out):
        # 取出反向传播的梯度
        if isinstance(grad_out, tuple):
            self.gradients = grad_out[0]
        else:
            self.gradients = grad_out

    def close(self):
        self.hook_f.remove()
        self.hook_b.remove()


# ==========================================
# 2. 模型与处理逻辑
# ==========================================
def load_qwen2_vl_model():
    print("Loading model from local directory... (Requires large VRAM for gradients)")
    model_path = "/root/autodl-tmp/Qwen2-VL-2B-Instruct"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到本地模型: {model_path}")

    # 注意：反向传播不能使用过于极限的量化，这里使用 bfloat16 或 float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Local Model loaded successfully!")
    return model, processor

def get_target_layer(model):
    """动态遍历模型结构，自动寻找视觉层的最后一层 Block"""
    for name, module in model.named_modules():
        # 寻找名字里带 'visual' 或 'vision' 的模块
        if 'visual' in name or 'vision' in name:
            # 检查这个模块里是否有堆叠的 blocks 或 layers
            if hasattr(module, 'blocks') and isinstance(module.blocks, torch.nn.ModuleList) and len(module.blocks) > 0:
                return module.blocks[-1]
            if hasattr(module, 'layers') and isinstance(module.layers, torch.nn.ModuleList) and len(module.layers) > 0:
                return module.layers[-1]
    
    # 如果实在找不到，打印出模型结构供人工排查
    print(model)
    raise ValueError("自动寻找视觉层失败！请检查上方打印的模型结构。")

def process_single_video_gradcam(model, processor, video_path, prompt, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path}, 
        {"type": "text", "text": prompt}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # ------------------------------------------------------------------
    # 阶段一：纯前向推理，获取模型预测的答案 Token
    # ------------------------------------------------------------------
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10, 
            use_cache=True,
            return_dict_in_generate=True
        )
    
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[0][input_len:]
    ans_text = processor.decode(generated_ids, skip_special_tokens=True)
    print(f"  -> Model Answer: {ans_text}")
    
    # 假设我们对它生成的第一个词（通常是选项字母）求导
    target_token_id = generated_ids[0].item()

    # ------------------------------------------------------------------
    # 阶段二：绑定 Hook，准备反向传播
    # ------------------------------------------------------------------
    # 动态获取视觉塔的最后一层 Block
    target_layer = get_target_layer(model)
    hook = HookBase(target_layer)

    # 强制模型开启梯度图
# 强制模型开启梯度图，但进行极限显存优化
    model.eval() 
    model.zero_grad()
    
    # 1. 先把全模型的梯度统统关掉 (极限省显存)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 只把“视觉塔 (Vision Tower)”的梯度打开
    for name, module in model.named_modules():
        if name == 'visual' or name == 'vision_model':
            for param in module.parameters():
                param.requires_grad = True
            break
            
    # 3. 确保我们要挂 Hook 的那一层绝对开启了梯度
    for param in target_layer.parameters():
        param.requires_grad = True

    # 仅将需要求导的图开启计算
    with torch.set_grad_enabled(True):
        forward_outputs = model(**inputs)
        logits = forward_outputs.logits
        
        # 提取模型对预测的下一个词（我们选定的 target_token_id）的置信度 logit
        target_logit = logits[0, -1, target_token_id]

        print(f"  -> Computing Gradients for token [{processor.decode([target_token_id])}]... Please wait.")
        target_logit.backward() # 核心：反向传播！触发 Hook！

    # ------------------------------------------------------------------
    # 阶段三：从 Hook 中提取特征与梯度，计算 Grad-CAM
    # ------------------------------------------------------------------
    features = hook.features.detach()   
    gradients = hook.gradients.detach() 

    # 获取视频的网格尺寸 [Time, Height, Width]
    t_grid, h_grid, w_grid = inputs['video_grid_thw'][0].tolist()
    h_tokens, w_tokens = h_grid // 2, w_grid // 2
    num_vision_tokens = t_grid * h_tokens * w_tokens

    if len(features.shape) == 3:
        features = features[0]
        gradients = gradients[0]

    # 截取纯视觉部分的 Token
    features = features[:num_vision_tokens].view(t_grid, h_tokens, w_tokens, -1)
    gradients = gradients[:num_vision_tokens].view(t_grid, h_tokens, w_tokens, -1)

    # Grad-CAM 核心公式：
    # 1. 对梯度求全局平均池化 (GAP)，得到通道权重 alpha
    weights = torch.mean(gradients, dim=(0, 1, 2), keepdim=True)

    # 2. 将权重与特征图相乘加和
    cam = torch.sum(weights * features, dim=-1)
    
    # 3. ReLU 激活：只关注对目标分类有正向影响的区域
    # 先转成标准的 float32，再交给 NumPy
    cam = F.relu(cam).to(torch.float32).cpu().numpy()

    # ------------------------------------------------------------------
    # 阶段四：叠加热力图到原始视频帧
    # ------------------------------------------------------------------
    frames = video_inputs[0]
    for t_idx in range(t_grid):
        frame_cam = cam[t_idx]
        
        if np.max(frame_cam) > 0:
            frame_cam = frame_cam - np.min(frame_cam)
            frame_cam = frame_cam / np.max(frame_cam)
        else:
            frame_cam = np.zeros_like(frame_cam)

        raw_frame = frames[t_idx]
        if hasattr(raw_frame, 'cpu'): raw_frame = raw_frame.cpu().numpy()
        if len(raw_frame.shape) == 3 and raw_frame.shape[0] == 3: raw_frame = raw_frame.transpose(1, 2, 0)
        if raw_frame.dtype != np.uint8:
            raw_frame = (raw_frame * 255).astype(np.uint8) if raw_frame.max() <= 1.0 else raw_frame.astype(np.uint8)

        target_h, target_w = raw_frame.shape[0], raw_frame.shape[1]
        
        cam_resized = cv2.resize(frame_cam, (target_w, target_h))
        heatmap = np.uint8(255 * cam_resized)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(raw_frame, 0.6, heatmap_colored, 0.4, 0)
        
        save_path = os.path.join(save_dir, f'frame_{t_idx:03d}_gradcam.jpg')
        cv2.imwrite(save_path, superimposed_img)

    print(f"  -> Grad-CAM completely saved for {t_grid} frames.")

    # ------------------------------------------------------------------
    # 阶段五：清理现场，防止显存泄漏
    # ------------------------------------------------------------------
    hook.close()
    for param in model.parameters():
        param.requires_grad = False
    del inputs, outputs, forward_outputs, logits, features, gradients, frames, cam
    torch.cuda.empty_cache()
    gc.collect()


# ==========================================
# 3. 主函数执行逻辑
# ==========================================
def main():
    base_dir = "/root/autodl-tmp/TAM-main"
    video_dir = os.path.join(base_dir, "MVBench/video/star/Charades_v1_480")
    json_path = os.path.join(base_dir, "MVBench/action_sequence.json")
    output_dir = os.path.join(base_dir, "result_gradcam")

    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    model, processor = load_qwen2_vl_model()

    target_data = qa_data[0:2]
    total_videos = len(target_data)

    for index, item in enumerate(target_data):
        video_filename = item.get("video")
        video_path = os.path.join(video_dir, video_filename)

        if not os.path.exists(video_path):
            continue

        question = item.get("question", "")
        candidates = item.get("candidates", [])
        
        options_text = ""
        for i, cand in enumerate(candidates):
            options_text += f"({chr(ord('A') + i)}) {cand}\n"

        prompt = f"Carefully watch the video and answer the following question.\nQuestion: {question}\nOptions:\n{options_text}\nPlease output the correct option letter only."

        video_name_without_ext = os.path.splitext(video_filename)[0]
        video_save_dir = os.path.join(output_dir, video_name_without_ext)

        print(f"\n[{index+1}/{total_videos}] Processing Video: {video_filename}")
        process_single_video_gradcam(model, processor, video_path, prompt, video_save_dir)
        
    print("\nAll target videos processed with Grad-CAM successfully!")

if __name__ == "__main__":
    main()