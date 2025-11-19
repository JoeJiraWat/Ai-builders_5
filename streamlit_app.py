import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import io
import os

# --- 1. MODEL ARCHITECTURE (ต้องเหมือนกับตอน Train เป๊ะๆ) ---
class RVC_AnimeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # โครงสร้างต้องตรงกับไฟล์ train.py ทุกประการ
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 256, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 5, padding=2), nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, 5, padding=2), nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

# --- 2. CONFIG & UTILS ---
CONFIG = {
    "sample_rate": 24000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "model/rvc_anime_epoch_50.pth"
}

@st.cache_resource
def load_model():
    """โหลดโมเดลเข้า Memory แค่ครั้งเดียว"""
    device = torch.device(CONFIG['device'])
    model = RVC_AnimeModel().to(device)
    
    # เช็คว่ามีไฟล์โมเดลไหม
    if not os.path.exists(CONFIG['model_path']):
        return None, f"❌ ไม่พบไฟล์โมเดลที่ {CONFIG['model_path']}"
    
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        # โหลดเฉพาะ state_dict ของโมเดล (ตัดพวก optimizer ทิ้งไป)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint) # เผื่อกรณี save แบบเก่า
            
        model.eval() # สำคัญ! ปิด Dropout/BatchNorm เพื่อเตรียมใช้งานจริง
        return model, "✅ โหลดโมเดลสำเร็จพร้อมใช้งาน!"
    except Exception as e:
        return None, f"❌ เกิดข้อผิดพลาด: {e}"

def process_audio(uploaded_file, model):
    """ฟังก์ชันแปลงเสียง"""
    device = torch.device(CONFIG['device'])
    
    # 1. อ่านไฟล์เสียงจาก Memory
    waveform, sr = torchaudio.load(uploaded_file)
    
    # 2. Resample ให้ตรงกับตอนเทรน (24kHz)
    if sr != CONFIG['sample_rate']:
        resampler = T.Resample(sr, CONFIG['sample_rate'])
        waveform = resampler(waveform)
        
    # 3. Convert to Mono (ถ้ามี 2 channel รวมให้เหลือ 1)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 4. ส่งเข้าโมเดล (Inference)
    input_tensor = waveform.unsqueeze(0).to(device) # เพิ่ม Batch dim -> [1, 1, Length]
    
    with torch.no_grad(): # ไม่ต้องคำนวณ Gradient ประหยัด RAM
        output_tensor = model(input_tensor)
        
    # 5. แปลงกลับเป็น Audio File
    output_waveform = output_tensor.squeeze(0).cpu() # ตัด Batch dim ออก
    
    # Save ลง Buffer (Virtual File) เพื่อส่งกลับไปหน้าเว็บ
    buffer = io.BytesIO()
    torchaudio.save(buffer, output_waveform, CONFIG['sample_rate'], format="wav")
    buffer.seek(0)
    
    return buffer

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Anime Voice Converter", page_icon="🎤")

st.title("🎤 Anime Voice Changer (RVC Demo)")
st.markdown("แปลงเสียงพูดของคุณให้กลายเป็นเสียง Anime ด้วย AI")

# Sidebar
st.sidebar.header("Model Status")
model, status_msg = load_model()
if model:
    st.sidebar.success(status_msg)
else:
    st.sidebar.error(status_msg)
    st.stop() # หยุดทำงานถ้าโหลดโมเดลไม่ได้

# Main Area
st.subheader("1. Upload Your Voice")
uploaded_file = st.file_uploader("เลือกไฟล์เสียง (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # แสดงเสียงต้นฉบับ
    st.audio(uploaded_file, format='audio/wav')
    st.info(f"Original File: {uploaded_file.name}")
    
    if st.button("✨ แปลงเสียงเป็น Anime ✨", type="primary"):
        with st.spinner('AI กำลังร่ายเวทมนตร์... (Converting)'):
            try:
                # ทำการแปลงเสียง
                converted_audio_bytes = process_audio(uploaded_file, model)
                
                st.success("เสร็จเรียบร้อย! (Done)")
                
                st.subheader("2. Result (Anime Voice)")
                # แสดงเสียงที่แปลงแล้ว
                st.audio(converted_audio_bytes, format='audio/wav')
                
                # ปุ่มดาวน์โหลด
                st.download_button(
                    label="📥 ดาวน์โหลดไฟล์เสียงใหม่",
                    data=converted_audio_bytes,
                    file_name="anime_voice_converted.wav",
                    mime="audio/wav"
                )
                
                # แสดงกราฟเทียบเล่นๆ (Optional)
                st.markdown("---")
                st.caption("Waveform Visualization")
                st.image("https://upload.wikimedia.org/wikipedia/commons/c/c5/Waveform_sine_wave.png", width=300) # ใส่รูปหลอกๆ หรือจะเขียน code plot กราฟจริงก็ได้
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดขณะแปลงเสียง: {e}")

st.markdown("---")
st.caption("Powered by PyTorch & Streamlit | Model: Simple Conv1d RVC")