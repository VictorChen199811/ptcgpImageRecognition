
# PTCGP卡牌識別系統

## 專案簡介
這是一個基於電腦視覺的寶可夢卡牌識別系統，能夠自動識別和分類寶可夢卡牌圖片。系統使用FastAPI建構後端服務，支援批量卡牌識別。

## 主要功能
- 自動檢測圖片中的卡牌位置
- 支援同時識別多張卡牌（最多5張）
- 使用圖像哈希算法進行卡牌匹配
- RESTful API 接口支援
- 支援多種圖片格式（PNG、JPG、JPEG、WEBP）

## 技術特點
- 使用 OpenCV 進行圖像處理和卡牌檢測
- 採用 perceptual hash 演算法進行圖像匹配
- FastAPI 提供高效能的 API 服務
- 支援異步文件處理

## 安裝說明
1. 克隆專案

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. 安裝依賴

```bash
pip install -r requirements.txt
```

## 使用方法
1. 運行主程序

```bash
uvicorn app.main:app --reload 
```

後打開瀏覽器進入 http://127.0.0.1:8000/docs#/default

2. 使用參考模型

```python
from models.reference_model import ReferenceModel
model = ReferenceModel()
model.train(data)
predictions = model.predict(test_data)
```

## 專案結構

```
.
├── app/
│   ├── main.py
│   └── reference_model.py
├── data/
├── models/
│   └── reference_model.py
├── README.md
└── requirements.txt
```

## 配置要求
- Python 3.8+
- 相關依賴包（詳見 requirements.txt）

## 貢獻指南
1. Fork 本專案
2. 創建新的分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 授權協議
本專案採用 MIT 授權協議 - 查看 [LICENSE](LICENSE) 文件了解更多詳情

## 聯繫方式
- 項目維護者：阿拉伯企鵝
- 電子郵件：lujiang.scout@gmail.com

## 致謝
感謝所有對本專案做出貢獻的開發者。
