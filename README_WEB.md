# Intelligent Stock Advisory System (Web)

Hướng dẫn chạy nhanh frontend (React/Vite) và backend (FastAPI).

## Yêu cầu
- Node.js + pnpm (hoặc npm/yarn nếu muốn)
- Python 3.10+ (để tạo venv)

## Cài đặt dependencies
### Frontend/Workspace
```bash
# từ thư mục web/
pnpm install
# (hoặc npm install / yarn install nếu không dùng pnpm)
```

### Backend
```bash
cd server
python -m venv venv
# Windows CMD/PowerShell
venv\Scripts\activate
# Windows Git Bash
source venv/Scripts/activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

## Chạy ứng dụng ở chế độ dev

Chạy cả client và server cùng lúc
```bash
cd web
pnpm dev
```

Hoặc mở 2 terminal riêng:

### 1) Backend (FastAPI)
```bash
cd web/server
# kích hoạt venv như trên
uvicorn main:app --reload --port 5000
```
API mặc định: http://localhost:5000

### 2) Frontend (React/Vite)
```bash
cd web/client
pnpm dev --host --port 5173
```
Frontend dev: http://localhost:5173

## Kiểm tra nhanh
- Mở frontend: http://localhost:5173
- Gọi API thử: http://localhost:5000/top100-list và http://localhost:5000/predict-top100-csv

## Build production (tuỳ chọn)
```bash
cd web/client
pnpm build
pnpm preview --host --port 4173
```
Preview: http://localhost:4173

## Ghi chú
- Cần file `server/top100_predictions.csv` nếu dùng endpoint `/predict-top100-csv` mà không chạy dự đoán live.
- Khi đổi cổng/backend URL, cập nhật cấu hình gọi API trong client (services/api.js).
