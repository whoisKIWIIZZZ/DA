import torch
import clip
from clip import load
import numpy as np
# DOWNLOAD_ROOT 保持不变
DOWNLOAD_ROOT='/home/yishouzhuo/work/sfda/clip-module'

from PIL import Image

def check_image_match_fixed(image_path):
    # 路径验证（确保文件存在，实际运行时请确保路径正确）
    try:
        Image.open(image_path)
    except FileNotFoundError:
        print(f"错误：未找到图片文件 {image_path}。请检查路径。")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载模型和预处理
    model, _, preprocess = load("ViT-B/32", device=device, download_root=DOWNLOAD_ROOT)

    # 图片处理
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 文本处理：我们增加一个对比项来让 Softmax 有意义，
    # 但主要目标是看 Logit。
    texts = ["a photo of a cat",  # 正向对比项
             "a photo of a dog",
             "a photo of not cat",
             "a photo unrelated to dog",
             "a photo where dog is missing"] # 你的目标项
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # 归一化 (L2)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 1. 计算余弦相似度 (Cosine Similarity) [-1, 1]
        cosine_similarity = image_features @ text_features.T
        
        # 2. 计算 Logit (Raw Logit)
        # CLIP 内部的温度 tau 通常是 100.0
        logits = 100.0 * cosine_similarity

    print("--- CLIP 相似度分析 (Dog 图像) ---")
    print("-----------------------------------")
    for i, text in enumerate(texts):
        sim = cosine_similarity[0][i].item()
        logit = logits[0][i].item()
        
        print(f"文本: \"{text}\"")
        print(f"  > Cosine Similarity: {sim:.4f}")
        print(f"  > Raw Logit (Tau=100): {logit:.2f}")
    
    # 如果 Softmax 仍需要，在这里加上
    # probs = torch.softmax(logits, dim=-1)
    # print("\nSoftmax 概率 (Sum=1.00):")
    # for i, text in enumerate(texts):
    #     print(f"  > {text}: {probs[0][i].item()*100:.2f}%")


# 注意：请替换成您实际的图片路径
check_image_match_fixed("/home/yishouzhuo/work/sfda/dog.jpeg")