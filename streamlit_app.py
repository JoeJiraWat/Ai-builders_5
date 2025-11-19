import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import io
import os
import numpy as np

# Import soundfile with error handling
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None  # Set to None to avoid NameError

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
    """โหลดโมเดลเข้า Memory แค่ครั้งเดียว (Cache ไว้ไม่ต้องโหลดซ้ำ)"""
    device = torch.device(CONFIG['device'])
    
    # เช็คว่ามีไฟล์โมเดลไหม
    if not os.path.exists(CONFIG['model_path']):
        return None, f"❌ ไม่พบไฟล์โมเดลที่ {CONFIG['model_path']}"
    
    try:
        # สร้างโมเดล
        model = RVC_AnimeModel()
        
        # โหลด weights (ใช้ weights_only=True เพื่อความปลอดภัยและเร็วขึ้น)
        try:
            checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
        except TypeError:
            # ถ้า PyTorch version เก่าไม่รองรับ weights_only
            checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        
        # โหลดเฉพาะ state_dict ของโมเดล (ตัดพวก optimizer ทิ้งไป)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False) # เผื่อกรณี save แบบเก่า
        
        # ย้ายโมเดลไป device และตั้งเป็น eval mode
        model = model.to(device)
        model.eval() # สำคัญ! ปิด Dropout/BatchNorm เพื่อเตรียมใช้งานจริง
        
        # ปิด gradient เพื่อประหยัด memory
        for param in model.parameters():
            param.requires_grad = False
            
        return model, "✅ โหลดโมเดลสำเร็จพร้อมใช้งาน!"
    except Exception as e:
        return None, f"❌ เกิดข้อผิดพลาด: {e}"

def process_audio(uploaded_file, model):
    """ฟังก์ชันแปลงเสียง"""
    device = torch.device(CONFIG['device'])
    
    try:
        # 1. อ่านไฟล์เสียงจาก Memory
        # ตรวจสอบว่าเป็น BytesIO หรือ file object
        if hasattr(uploaded_file, 'read'):
            # ถ้าเป็น file object ให้ reset pointer
            uploaded_file.seek(0)
        
        # ใช้ soundfile ถ้ามี (ไม่ต้องใช้ torchcodec) หรือใช้ torchaudio
        if SOUNDFILE_AVAILABLE:
            try:
                # ลองใช้ soundfile ก่อน (รองรับไฟล์ส่วนใหญ่และไม่ต้องใช้ torchcodec)
                audio_data, sr = sf.read(uploaded_file, dtype='float32', always_2d=False)
                
                # ตรวจสอบว่าได้ข้อมูลเสียงหรือไม่
                if audio_data.size == 0:
                    raise ValueError("ไฟล์เสียงว่างเปล่า")
                
                # แปลงเป็น torch tensor
                if len(audio_data.shape) == 1:
                    # Mono audio
                    waveform = torch.from_numpy(audio_data).unsqueeze(0)  # [1, length]
                else:
                    # Multi-channel audio
                    waveform = torch.from_numpy(audio_data.T)  # [channels, length]
                    
            except Exception as e:
                # ถ้า soundfile ไม่รองรับ ลองใช้ torchaudio
                try:
                    # Reset file pointer
                    if hasattr(uploaded_file, 'seek'):
                        uploaded_file.seek(0)
                    waveform, sr = torchaudio.load(uploaded_file)
                    if waveform.numel() == 0:
                        raise ValueError("ไฟล์เสียงว่างเปล่า")
                except Exception as e2:
                    raise Exception(f"ไม่สามารถโหลดไฟล์เสียงได้ (soundfile error: {str(e)}, torchaudio error: {str(e2)})")
        else:
            # ใช้ torchaudio โดยตรง (อาจต้องใช้ torchcodec)
            try:
                # Reset file pointer
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
                waveform, sr = torchaudio.load(uploaded_file)
                if waveform.numel() == 0:
                    raise ValueError("ไฟล์เสียงว่างเปล่า")
            except Exception as e:
                raise Exception(f"ไม่สามารถโหลดไฟล์เสียงได้: {str(e)}. กรุณาติดตั้ง soundfile หรือ torchcodec")
        
        # 2. Convert to Mono (ถ้ามี 2 channel รวมให้เหลือ 1)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # ตรวจสอบว่า waveform มีข้อมูลหรือไม่
        if waveform.numel() == 0:
            raise ValueError("ไม่สามารถประมวลผลเสียงได้ - ข้อมูลเสียงว่างเปล่า")
        
        # 3. Resample ให้ตรงกับตอนเทรน (24kHz)
        if sr != CONFIG['sample_rate']:
            try:
                resampler = T.Resample(sr, CONFIG['sample_rate']).to(device)
                waveform = waveform.to(device)
                waveform = resampler(waveform)
            except Exception as e:
                raise Exception(f"ไม่สามารถ resample เสียงได้: {str(e)}")
        else:
            waveform = waveform.to(device)
            
        # ตรวจสอบว่า waveform ยังมีข้อมูลหรือไม่หลัง resample
        if waveform.numel() == 0:
            raise ValueError("ไม่สามารถประมวลผลเสียงได้ - ข้อมูลเสียงหายไปหลัง resample")
        
        # 4. ส่งเข้าโมเดล (Inference)
        input_tensor = waveform.unsqueeze(0) # เพิ่ม Batch dim -> [1, 1, Length]
        
        try:
            with torch.no_grad(): # ไม่ต้องคำนวณ Gradient ประหยัด RAM
                output_tensor = model(input_tensor)
        except Exception as e:
            raise Exception(f"เกิดข้อผิดพลาดขณะประมวลผลด้วยโมเดล: {str(e)}")
        
        # ตรวจสอบผลลัพธ์
        if output_tensor.numel() == 0:
            raise ValueError("โมเดลไม่สามารถแปลงเสียงได้ - ผลลัพธ์ว่างเปล่า")
        
        # 5. แปลงกลับเป็น Audio File
        output_waveform = output_tensor.squeeze(0).cpu() # ตัด Batch dim ออก
        
        # Normalize audio เพื่อป้องกัน clipping
        max_val = torch.abs(output_waveform).max()
        if max_val > 0:
            output_waveform = output_waveform / max_val * 0.95  # ป้องกัน clipping
        
        # Save ลง Buffer (Virtual File) เพื่อส่งกลับไปหน้าเว็บ
        buffer = io.BytesIO()
        try:
            torchaudio.save(buffer, output_waveform, CONFIG['sample_rate'], format="wav")
            buffer.seek(0)
        except Exception as e:
            raise Exception(f"ไม่สามารถบันทึกไฟล์เสียงได้: {str(e)}")
        
        return buffer
        
    except Exception as e:
        # Re-raise exception พร้อมข้อมูลเพิ่มเติม
        error_msg = f"เกิดข้อผิดพลาดในการแปลงเสียง: {str(e)}"
        raise Exception(error_msg)

# --- 3. STREAMLIT UI ---
st.set_page_config(
    page_title="Anime Voice Converter", 
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# แสดง loading indicator ก่อนโหลดโมเดล
with st.spinner('🔄 กำลังโหลดโมเดล AI... กรุณารอสักครู่'):
    model, status_msg = load_model()

st.title("🎤 Anime Voice Changer (RVC Demo)")
st.markdown("แปลงเสียงพูดของคุณให้กลายเป็นเสียง Anime ด้วย AI")

# Sidebar
st.sidebar.header("Model Status")
if model:
    st.sidebar.success(status_msg)
    st.sidebar.info(f"⚙️ Device: {CONFIG['device'].upper()}")
else:
    st.sidebar.error(status_msg)
    st.stop() # หยุดทำงานถ้าโหลดโมเดลไม่ได้

# Main Area
st.subheader("1. เลือกวิธีบันทึกเสียง")

# สร้าง tabs สำหรับเลือกวิธี
tab1, tab2 = st.tabs(["🎙️ บันทึกเสียงตรงนี้", "📁 อัปโหลดไฟล์"])

audio_source = None
audio_name = None

with tab1:
    st.markdown("### 🎤 กดปุ่มด้านล่างเพื่อบันทึกเสียงของคุณ")
    st.info("💡 **คำแนะนำ**: กดปุ่มแล้วพูดเสียงที่ต้องการแปลง จากนั้นกดหยุดบันทึก\n\n⚠️ **หมายเหตุ**: ต้องอนุญาตให้เว็บไซต์เข้าถึงไมโครโฟน")
    
    # ฟีเจอร์บันทึกเสียง
    try:
        audio_bytes = st.audio_input("บันทึกเสียง", label_visibility="collapsed")
        
        if audio_bytes is not None:
            # แปลง audio_bytes เป็น BytesIO object
            audio_source = io.BytesIO(audio_bytes)
            audio_name = "recorded_audio.wav"
            st.success("✅ บันทึกเสียงสำเร็จ!")
            st.audio(audio_bytes, format='audio/wav')
            st.info("🎵 เสียงที่บันทึกแล้ว - พร้อมแปลงเป็น Anime Voice!")
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถบันทึกเสียงได้: {e}")
        st.info("💡 กรุณาลองใช้วิธีอัปโหลดไฟล์แทน หรือตรวจสอบการอนุญาตไมโครโฟน")

with tab2:
    st.markdown("### 📁 อัปโหลดไฟล์เสียงจากเครื่อง")
    uploaded_file = st.file_uploader("เลือกไฟล์เสียง (.wav, .mp3)", type=["wav", "mp3"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        audio_source = uploaded_file
        audio_name = uploaded_file.name
        st.success("✅ อัปโหลดไฟล์สำเร็จ!")
        st.audio(uploaded_file, format='audio/wav')
        st.info(f"📄 ไฟล์: {uploaded_file.name}")

# แสดงปุ่มแปลงเสียงเมื่อมีเสียงแล้ว
if audio_source is not None:
    st.markdown("---")
    st.subheader("2. แปลงเสียง")
    
    # แสดงเสียงต้นฉบับ
    st.audio(audio_source, format='audio/wav')
    st.caption(f"🎵 เสียงต้นฉบับ: {audio_name}")
    
    # ปุ่มแปลงเสียง
    if st.button("✨ แปลงเสียงเป็น Anime ✨", type="primary", use_container_width=True):
        try:
            with st.spinner('🤖 AI กำลังร่ายเวทมนตร์... (Converting)'):
                # Reset file pointer เพื่อให้อ่านได้ใหม่
                if hasattr(audio_source, 'seek'):
                    audio_source.seek(0)
                
                # ตรวจสอบว่าโมเดลพร้อมหรือไม่
                if model is None:
                    st.error("❌ โมเดลยังไม่พร้อมใช้งาน")
                    st.stop()
                
                # ทำการแปลงเสียง
                converted_audio_bytes = process_audio(audio_source, model)
                
                # ตรวจสอบว่าได้ผลลัพธ์หรือไม่
                if converted_audio_bytes is None:
                    st.error("❌ ไม่สามารถแปลงเสียงได้ - ไม่มีผลลัพธ์")
                    st.stop()
                
                st.success("✅ เสร็จเรียบร้อย! (Done)")
                
                st.subheader("🎉 ผลลัพธ์ (Anime Voice)")
                # แสดงเสียงที่แปลงแล้ว
                try:
                    st.audio(converted_audio_bytes, format='audio/wav')
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถแสดงเสียงได้: {e}")
                
                # ปุ่มดาวน์โหลด
                try:
                    st.download_button(
                        label="📥 ดาวน์โหลดไฟล์เสียงใหม่",
                        data=converted_audio_bytes,
                        file_name="anime_voice_converted.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถสร้างปุ่มดาวน์โหลดได้: {e}")
                    
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดขณะแปลงเสียง")
            st.error(f"**รายละเอียด**: {str(e)}")
            
            # แสดงข้อมูลเพิ่มเติมสำหรับ debugging
            with st.expander("🔍 ดูรายละเอียดข้อผิดพลาด"):
                st.exception(e)
            
            st.info("💡 **คำแนะนำ**:")
            st.info("1. ตรวจสอบว่าไฟล์เสียงไม่เสียหาย")
            st.info("2. ลองใช้ไฟล์เสียงอื่น")
            st.info("3. ตรวจสอบว่าไฟล์เสียงมีขนาดไม่ใหญ่เกินไป")

st.markdown("---")
st.caption("Powered by PyTorch & Streamlit | Model: Simple Conv1d RVC")