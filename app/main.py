from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from models import reference_model

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    try:
        # 保存上傳的文件
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"文件已保存到: {file_path}")
        
        # 處理圖片
        results = reference_model.main(file_path)
        
        matches = [result["match"] for result in results]
        
        return {
            "status": "success",
            "results": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理臨時文件
        if os.path.exists(file_path):
            os.remove(file_path)
        try:
            os.rmdir(temp_dir)
        except:
            pass