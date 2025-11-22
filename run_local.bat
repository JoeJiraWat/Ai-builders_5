@echo off
REM Script สำหรับรัน Streamlit App แบบ Local (Windows)

echo 🚀 กำลังเริ่มต้น Streamlit App...
echo.

REM ตรวจสอบว่า streamlit ติดตั้งแล้วหรือยัง
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Streamlit ยังไม่ได้ติดตั้ง
    echo 📦 กำลังติดตั้ง dependencies...
    pip install -r requirements.txt
    echo.
)

REM ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
if not exist "model\rvc_anime_epoch_50.pth" (
    echo ⚠️  ไฟล์โมเดลไม่พบ: model\rvc_anime_epoch_50.pth
    echo    กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้อง
    echo.
)

REM รัน Streamlit
echo ✅ กำลังเปิด Streamlit App...
echo 🌐 เปิดเบราว์เซอร์ไปที่: http://localhost:8501
echo.
echo 💡 กด Ctrl+C เพื่อหยุดการทำงาน
echo.

streamlit run streamlit_app.py --server.port 8501 --server.address localhost

pause

