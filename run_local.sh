#!/bin/bash

# Script สำหรับรัน Streamlit App แบบ Local

echo "🚀 กำลังเริ่มต้น Streamlit App..."
echo ""

# ตรวจสอบว่า streamlit ติดตั้งแล้วหรือยัง
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit ยังไม่ได้ติดตั้ง"
    echo "📦 กำลังติดตั้ง dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
if [ ! -f "model/rvc_anime_epoch_50.pth" ]; then
    echo "⚠️  ไฟล์โมเดลไม่พบ: model/rvc_anime_epoch_50.pth"
    echo "   กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้อง"
    echo ""
fi

# รัน Streamlit
echo "✅ กำลังเปิด Streamlit App..."
echo "🌐 เปิดเบราว์เซอร์ไปที่: http://localhost:8501"
echo ""
echo "💡 กด Ctrl+C เพื่อหยุดการทำงาน"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost

