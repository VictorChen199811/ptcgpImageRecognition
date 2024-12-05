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
    
    # 圖像預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny邊緣檢測
    edges = cv2.Canny(blurred, 50, 150)
    
    # 使用形態學操作
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 查找輪廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"找到 {len(contours)} 個輪廓")
    
    card_regions = []
    min_area = image.shape[0] * image.shape[1] * 0.05  # 增加最小面積閾值
    max_area = image.shape[0] * image.shape[1] * 0.25  # 調整最大面積閾值
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # 計算長寬比
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # 遊戲中卡片的長寬比範圍
        if 0.6 <= aspect_ratio <= 0.85:
            card_regions.append((x, y, w, h))
    
    print(f"檢測到 {len(card_regions)} 個可能的卡牌區域")
    
    # 根據位置排序（從上到下，從左到右）
    card_regions.sort(key=lambda x: (x[1] // (image.shape[0] // 3), x[0]))
    
    # 保存調試圖像
    debug_image = image.copy()
    for i, (x, y, w, h) in enumerate(card_regions):
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_image, str(i+1), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('debug_detection.jpg', debug_image)
    
    return card_regions

def remove_overlapping_regions(regions, overlap_thresh=0.3):
    """
    移除重疊的區域
    :param regions: 區域列表 [(x, y, w, h), ...]
    :param overlap_thresh: 重疊閾值
    :return: 過濾後的區域列表
    """
    if not regions:
        return []
        
    # 計算每個區域的面積
    areas = [w * h for (x, y, w, h) in regions]
    
    # 根據面積排序（從大到小）
    idxs = np.argsort(areas)[::-1]
    
    # 保存要保留的區域
    keep = []
    
    while len(idxs) > 0:
        # 保留最大的區域
        current = idxs[0]
        keep.append(current)
        
        # 計算當前區域與其他區域的重疊度
        overlapping = []
        for idx in idxs[1:]:
            overlap = calculate_overlap(regions[current], regions[idx])
            if overlap > overlap_thresh:
                overlapping.append(idx)
                
        # 移除重疊的區域
        idxs = [idx for idx in idxs[1:] if idx not in overlapping]
    
    return [regions[i] for i in keep]

def calculate_overlap(region1, region2):
    """
    計算兩個區域的重疊度
    """
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2
    
    # 計算相交區域
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    
    return intersection_area / float(min(area1, area2))


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
