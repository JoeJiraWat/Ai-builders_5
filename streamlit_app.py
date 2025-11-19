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
    sf = None

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class RVC_AnimeModel(nn.Module):
    """โมเดล RVC สำหรับแปลงเสียงเป็น Anime Voice"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 256, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 5, padding=2),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, 5, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "sample_rate": 24000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "model/rvc_anime_epoch_50.pth"
}

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """โหลดโมเดลและ cache ไว้ใน memory"""
    device = torch.device(CONFIG['device'])
    
    # ตรวจสอบไฟล์โมเดล
    if not os.path.exists(CONFIG['model_path']):
        return None, f"❌ ไม่พบไฟล์โมเดลที่ {CONFIG['model_path']}"
    
    try:
        # สร้างโมเดล
        model = RVC_AnimeModel()
        
        # โหลด weights
        try:
            checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        
        # โหลด state_dict
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # ตั้งค่าโมเดล
        model = model.to(device)
        model.eval()
        
        # ปิด gradient เพื่อประหยัด memory
        for param in model.parameters():
            param.requires_grad = False
            
        return model, "✅ โหลดโมเดลสำเร็จพร้อมใช้งาน!"
    except Exception as e:
        return None, f"❌ เกิดข้อผิดพลาด: {e}"

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def load_audio_file(audio_file):
    """โหลดไฟล์เสียงและแปลงเป็น torch tensor"""
    # Reset file pointer
    if hasattr(audio_file, 'seek'):
        audio_file.seek(0)
    
    # ใช้ soundfile ถ้ามี
    if SOUNDFILE_AVAILABLE:
        try:
            audio_data, sr = sf.read(audio_file, dtype='float32', always_2d=False)
            
            if audio_data.size == 0:
                raise ValueError("ไฟล์เสียงว่างเปล่า")
            
            # แปลงเป็น torch tensor
            if len(audio_data.shape) == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data.T)
            
            return waveform, sr
            
        except Exception as e:
            # Fallback ไปใช้ torchaudio
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)
            try:
                waveform, sr = torchaudio.load(audio_file)
                if waveform.numel() == 0:
                    raise ValueError("ไฟล์เสียงว่างเปล่า")
                return waveform, sr
            except Exception as e2:
                raise Exception(f"ไม่สามารถโหลดไฟล์เสียงได้: {str(e)} | {str(e2)}")
    else:
        # ใช้ torchaudio โดยตรง
        try:
            waveform, sr = torchaudio.load(audio_file)
            if waveform.numel() == 0:
                raise ValueError("ไฟล์เสียงว่างเปล่า")
            return waveform, sr
        except Exception as e:
            raise Exception(f"ไม่สามารถโหลดไฟล์เสียงได้: {str(e)}")

def preprocess_audio(waveform, sr, target_sr=24000):
    """ประมวลผลเสียงก่อนส่งเข้าโมเดล"""
    device = torch.device(CONFIG['device'])
    
    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr).to(device)
        waveform = waveform.to(device)
        waveform = resampler(waveform)
    else:
        waveform = waveform.to(device)
    
    return waveform

def process_audio(audio_file, model):
    """ฟังก์ชันหลักสำหรับแปลงเสียง"""
    device = torch.device(CONFIG['device'])
    
    try:
        # 1. โหลดไฟล์เสียง
        waveform, sr = load_audio_file(audio_file)
        
        # 2. Preprocess
        waveform = preprocess_audio(waveform, sr, CONFIG['sample_rate'])
        
        # ตรวจสอบข้อมูล
        if waveform.numel() == 0:
            raise ValueError("ข้อมูลเสียงว่างเปล่า")
        
        # 3. Inference
        input_tensor = waveform.unsqueeze(0)  # [1, 1, length]
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        if output_tensor.numel() == 0:
            raise ValueError("โมเดลไม่สามารถแปลงเสียงได้")
        
        # 4. Post-process
        output_waveform = output_tensor.squeeze(0).cpu()
        
        # Normalize เพื่อป้องกัน clipping
        max_val = torch.abs(output_waveform).max()
        if max_val > 0:
            output_waveform = output_waveform / max_val * 0.95
        
        # 5. บันทึกเป็นไฟล์
        buffer = io.BytesIO()
        torchaudio.save(buffer, output_waveform, CONFIG['sample_rate'], format="wav")
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการแปลงเสียง: {str(e)}")

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    """ฟังก์ชันหลักของ Streamlit App"""
    
    # Page config
    st.set_page_config(
        page_title="Anime Voice Converter",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # โหลดโมเดล
    with st.spinner('🔄 กำลังโหลดโมเดล AI... กรุณารอสักครู่'):
        model, status_msg = load_model()
    
    # Header
    st.title("🎤 Anime Voice Changer (RVC Demo)")
    st.markdown("แปลงเสียงพูดของคุณให้กลายเป็นเสียง Anime ด้วย AI")
    
    # Sidebar
    st.sidebar.header("Model Status")
    if model:
        st.sidebar.success(status_msg)
        st.sidebar.info(f"⚙️ Device: {CONFIG['device'].upper()}")
    else:
        st.sidebar.error(status_msg)
        st.stop()
    
    # Main content
    st.subheader("1. เลือกวิธีบันทึกเสียง")
    
    # Tabs
    tab1, tab2 = st.tabs(["🎙️ บันทึกเสียงตรงนี้", "📁 อัปโหลดไฟล์"])
    
    audio_source = None
    audio_name = None
    
    # Tab 1: บันทึกเสียง
    with tab1:
        st.markdown("### 🎤 กดปุ่มด้านล่างเพื่อบันทึกเสียงของคุณ")
        st.info("💡 **คำแนะนำ**: กดปุ่มแล้วพูดเสียงที่ต้องการแปลง จากนั้นกดหยุดบันทึก\n\n⚠️ **หมายเหตุ**: ต้องอนุญาตให้เว็บไซต์เข้าถึงไมโครโฟน")
        
        try:
            audio_bytes = st.audio_input("บันทึกเสียง", label_visibility="collapsed")
            
            if audio_bytes is not None:
                audio_source = io.BytesIO(audio_bytes)
                audio_name = "recorded_audio.wav"
                st.success("✅ บันทึกเสียงสำเร็จ!")
                st.audio(audio_bytes, format='audio/wav')
                st.info("🎵 เสียงที่บันทึกแล้ว - พร้อมแปลงเป็น Anime Voice!")
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถบันทึกเสียงได้: {e}")
            st.info("💡 กรุณาลองใช้วิธีอัปโหลดไฟล์แทน หรือตรวจสอบการอนุญาตไมโครโฟน")
    
    # Tab 2: อัปโหลดไฟล์
    with tab2:
        st.markdown("### 📁 อัปโหลดไฟล์เสียงจากเครื่อง")
        uploaded_file = st.file_uploader(
            "เลือกไฟล์เสียง (.wav, .mp3)",
            type=["wav", "mp3"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            audio_source = uploaded_file
            audio_name = uploaded_file.name
            st.success("✅ อัปโหลดไฟล์สำเร็จ!")
            st.audio(uploaded_file, format='audio/wav')
            st.info(f"📄 ไฟล์: {uploaded_file.name}")
    
    # แสดงปุ่มแปลงเสียง
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
                    # Reset file pointer
                    if hasattr(audio_source, 'seek'):
                        audio_source.seek(0)
                    
                    # ตรวจสอบโมเดล
                    if model is None:
                        st.error("❌ โมเดลยังไม่พร้อมใช้งาน")
                        st.stop()
                    
                    # แปลงเสียง
                    converted_audio_bytes = process_audio(audio_source, model)
                    
                    if converted_audio_bytes is None:
                        st.error("❌ ไม่สามารถแปลงเสียงได้ - ไม่มีผลลัพธ์")
                        st.stop()
                    
                    # แสดงผลลัพธ์
                    st.success("✅ เสร็จเรียบร้อย! (Done)")
                    st.subheader("🎉 ผลลัพธ์ (Anime Voice)")
                    
                    # แสดงเสียง
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
                st.error("❌ เกิดข้อผิดพลาดขณะแปลงเสียง")
                st.error(f"**รายละเอียด**: {str(e)}")
                
                # แสดงรายละเอียด error
                with st.expander("🔍 ดูรายละเอียดข้อผิดพลาด"):
                    st.exception(e)
                
                # คำแนะนำ
                st.info("💡 **คำแนะนำ**:")
                st.info("1. ตรวจสอบว่าไฟล์เสียงไม่เสียหาย")
                st.info("2. ลองใช้ไฟล์เสียงอื่น")
                st.info("3. ตรวจสอบว่าไฟล์เสียงมีขนาดไม่ใหญ่เกินไป")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by PyTorch & Streamlit | Model: Simple Conv1d RVC")

if __name__ == "__main__":
    main()
