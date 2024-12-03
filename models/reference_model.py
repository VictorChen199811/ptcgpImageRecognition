import os
import cv2
from PIL import Image
import imagehash
import numpy as np


def load_reference_data(data_dir, method="hash"):
    """
    加載參考圖像，支持哈希和 ORB 特徵兩種方法
    :param data_dir: 圖像根目錄
    :param method: "hash"
    :return: 參考圖像數據字典
    """
    if method == "hash":
        reference_data = {}
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                reference_data[category] = {}
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        try:
                            img = Image.open(file_path)
                            hash_value = imagehash.phash(img)
                            reference_data[category][file] = hash_value
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
        return reference_data

    elif method == "feature":
        orb = cv2.ORB_create()
        reference_data = {}
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                reference_data[category] = {}
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image = cv2.imread(file_path, 0)
                        kp, des = orb.detectAndCompute(image, None)
                        reference_data[category][file] = (kp, des)
        return reference_data


def match_image(target_image_path, reference_data, method="hash"):
    """
    匹配目標圖像
    :param target_image_path: 目標圖像路徑
    :param reference_data: 參考圖像數據
    :param method: "hash" 
    :return: 匹配結果
    """
    if method == "hash":
        target_hash = imagehash.phash(Image.open(target_image_path))
        best_match = None
        best_distance = float('inf')

        for category, images in reference_data.items():
            for file_name, ref_hash in images.items():
                distance = target_hash - ref_hash
                if distance < best_distance:
                    best_distance = distance
                    best_match = (category, file_name)

        return best_match, best_distance

    elif method == "feature":
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        target_image = cv2.imread(target_image_path, 0)
        kp_target, des_target = orb.detectAndCompute(target_image, None)

        best_match = None
        max_good_matches = 0

        for category, images in reference_data.items():
            for file_name, (kp_template, des_template) in images.items():
                matches = bf.match(des_template, des_target)
                good_matches = [m for m in matches if m.distance < 50]
                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_match = (category, file_name)

        return best_match, max_good_matches


def detect_cards(image_path):
    """
    檢測圖片中的卡牌位置
    :param image_path: 輸入圖片路徑
    :return: 卡牌區域列表 [(x, y, w, h), ...]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"無法讀取圖片: {image_path}")
        
    print(f"圖片尺寸: {image.shape}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 30, 150)
    
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"找到 {len(contours)} 個輪廓")
    
    card_regions = []
    min_area = image.shape[0] * image.shape[1] * 0.02  
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        if 0.6 <= aspect_ratio <= 0.8:
            card_regions.append((x, y, w, h))
    
    if len(card_regions) != 5:
        print(f"檢測到 {len(card_regions)} 個區域，進行後處理")
        
        if len(card_regions) > 5:
            card_regions = sorted(card_regions, key=lambda x: x[2] * x[3], reverse=True)[:5]
        
        if len(card_regions) < 5:
            print("使用固定分割方法")
            card_regions = []
            total_width = image.shape[1]
            total_height = image.shape[0]
            
            # 計算上排卡片的位置（3張）
            card_width = total_width // 3
            for i in range(3):
                x = i * card_width
                w = card_width
                h = int(total_height * 0.45)  
                card_regions.append((x, 0, w, h))
            
            # 計算下排卡片的位置（2張）
            card_width = total_width // 2
            y = int(total_height * 0.55)  
            h = total_height - y  
            for i in range(2):
                x = i * card_width + card_width // 4  
                w = card_width
                card_regions.append((x, y, w, h))
    
    # 根據位置排序（先上排從左到右，再下排從左到右）
    card_regions.sort(key=lambda x: (x[1], x[0]))
    
    print(f"最終檢測到 {len(card_regions)} 張卡牌")
    
    # 保存調試圖像
    debug_image = image.copy()
    for i, (x, y, w, h) in enumerate(card_regions):
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_image, str(i+1), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('debug_detection.jpg', debug_image)
    
    return card_regions


def extract_cards(image_path, output_dir="temp_cards"):
    """
    從圖片中提取卡牌
    :param image_path: 輸入圖片路徑
    :param output_dir: 輸出目錄
    :return: 卡牌圖片路徑列表
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    card_regions = detect_cards(image_path)

    card_paths = []
    for idx, (x, y, w, h) in enumerate(card_regions):
        # 提取卡牌圖像
        card = image[y:y + h, x:x + w]
        
        # 調整大小為統一尺寸
        card = cv2.resize(card, (360, 500))  # 標準寶可夢卡片比例
        
        # 保存卡牌圖像
        card_path = os.path.join(output_dir, f"card_{idx}.jpg")
        cv2.imwrite(card_path, card)
        card_paths.append(card_path)

    return card_paths


def main(target_image_path):
    data_dir = "data"
    temp_dir = "temp_cards"

    print(f"處理圖片: {target_image_path}")
    print("加載參考圖像...")
    reference_data = load_reference_data(data_dir)
    print(f"已加載 {sum(len(cat) for cat in reference_data.values())} 張參考圖像")

    print("檢測並提取卡牌...")
    try:
        card_paths = extract_cards(target_image_path, output_dir=temp_dir)
        print(f"提取出 {len(card_paths)} 張卡牌")
    except Exception as e:
        print(f"提取卡牌時發生錯誤: {e}")
        return []

    print("判斷卡牌...")
    results = []
    for idx, card_path in enumerate(card_paths):
        print(f"正在處理第 {idx+1} 張卡牌: {card_path}")
        try:
            match_result, score = match_image(card_path, reference_data)
            if match_result:
                category, file_name = match_result
                print(f"卡牌 {idx+1} 匹配結果: {category}/{file_name}, 分: {score}")
            else:
                print(f"卡牌 {idx+1} 沒有找到匹配")
            results.append({"card_index": idx+1, "match": match_result, "score": score})
        except Exception as e:
            print(f"處理卡牌 {idx+1} 時發生錯誤: {e}")
            results.append({"card_index": idx+1, "error": str(e)})

    # 清理臨時文件
    try:
        for card_path in card_paths:
            if os.path.exists(card_path):
                os.remove(card_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        print(f"清理臨時文件時發生錯誤: {e}")

    return results
