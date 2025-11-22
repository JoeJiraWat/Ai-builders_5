# 🚀 วิธีรัน Streamlit App แบบ Local

## 📋 ข้อกำหนดเบื้องต้น

1. Python 3.8 หรือสูงกว่า
2. pip (Python package manager)

## 🔧 วิธีติดตั้งและรัน

### วิธีที่ 1: ใช้ Script (แนะนำ)

**สำหรับ macOS/Linux:**
```bash
./run_local.sh
```

**สำหรับ Windows:**
```cmd
run_local.bat
```

### วิธีที่ 2: รันด้วยคำสั่งโดยตรง

1. **ติดตั้ง dependencies:**
```bash
pip install -r requirements.txt
```

2. **รัน Streamlit:**
```bash
streamlit run streamlit_app.py
```

หรือระบุ port:
```bash
streamlit run streamlit_app.py --server.port 8501
```

## 🌐 เข้าถึงแอพ

หลังจากรันคำสั่งแล้ว เปิดเบราว์เซอร์ไปที่:
- **URL:** http://localhost:8501
- Streamlit จะเปิดเบราว์เซอร์อัตโนมัติ

## ⚙️ การตั้งค่าเพิ่มเติม

### ใช้ GPU (ถ้ามี)

โค้ดจะตรวจสอบและใช้ GPU อัตโนมัติถ้ามี CUDA ติดตั้ง

### เปลี่ยน Port

แก้ไขใน script หรือใช้:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 🛑 หยุดการทำงาน

กด `Ctrl + C` ใน terminal

## ⚠️ หมายเหตุ

- ตรวจสอบว่าไฟล์โมเดล `model/rvc_anime_epoch_50.pth` อยู่ในตำแหน่งที่ถูกต้อง
- สำหรับไฟล์เสียงขนาดใหญ่ (>50MB) audio player อาจไม่แสดงได้ แต่ยังสามารถแปลงเสียงได้ปกติ

## 🐛 แก้ไขปัญหา

### ไม่สามารถรันได้
```bash
pip install --upgrade streamlit
pip install -r requirements.txt
```

### โมเดลไม่พบ
ตรวจสอบว่าไฟล์ `model/rvc_anime_epoch_50.pth` อยู่ในโฟลเดอร์ `model/`

### Port ถูกใช้งานแล้ว
เปลี่ยน port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

