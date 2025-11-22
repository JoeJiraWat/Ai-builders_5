import streamlit as st
import torch
import torch.nn as nn
import io
import os
import numpy as np
import traceback

# Import soundfile (หลัก)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDFILE_AVAILABLE = False
    sf = None

# Import torchaudio แบบ lazy (เมื่อจำเป็นจริงๆ)
TORCHAUDIO_AVAILABLE = False
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError) as e:
    # ถ้า torchaudio ไม่สามารถโหลดได้ ใช้ soundfile และ scipy แทน
    torchaudio = None
    T = None

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
def get_device():
    """ตรวจสอบและเลือก device ที่เหมาะสม"""
    # ตรวจสอบ CUDA
    if torch.cuda.is_available():
        device = "cuda"
        try:
            # ตรวจสอบว่า GPU ใช้งานได้จริง
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except Exception:
            # ถ้า GPU มีปัญหา ใช้ CPU แทน
            return "cpu"
    else:
        return "cpu"

CONFIG = {
    "sample_rate": 24000,
    "device": get_device(),
    "model_path": "model/rvc_anime_epoch_50.pth"
}

# ฟังก์ชันสำหรับแสดงข้อมูล GPU
def get_gpu_info():
    """ดึงข้อมูล GPU"""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return f"{gpu_name} ({gpu_memory:.1f} GB)"
        except:
            return "GPU Available (Unknown)"
    else:
        return "CPU Only"

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
        error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
        return None, error_msg

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def load_audio_file(audio_file):
    """โหลดไฟล์เสียงและแปลงเป็น torch tensor"""
    errors = []
    
    # Reset file pointer
    try:
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)
    except Exception as e:
        errors.append(f"ไม่สามารถ reset file pointer: {str(e)}")
    
    # วิธีที่ 1: ใช้ soundfile
    if SOUNDFILE_AVAILABLE:
        try:
            # สำหรับ BytesIO หรือ file-like object
            if hasattr(audio_file, 'read'):
                # ต้องอ่านเป็น bytes ก่อน
                audio_file.seek(0)
                audio_bytes = audio_file.read()
                temp_file = io.BytesIO(audio_bytes)
                audio_data, sr = sf.read(temp_file, dtype='float32', always_2d=False)
            else:
                audio_data, sr = sf.read(audio_file, dtype='float32', always_2d=False)
            
            if audio_data.size == 0:
                raise ValueError("ไฟล์เสียงว่างเปล่า")
            
            # แปลงเป็น torch tensor
            if len(audio_data.shape) == 1:
                waveform = torch.from_numpy(audio_data.copy()).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio_data.T.copy()).float()
            
            if waveform.numel() == 0:
                raise ValueError("ไม่สามารถโหลดข้อมูลเสียงได้")
            
            return waveform, int(sr)
            
        except Exception as e:
            errors.append(f"soundfile error: {str(e)}")
    
    # วิธีที่ 2: ใช้ torchaudio (fallback) - ถ้ามี
    if TORCHAUDIO_AVAILABLE:
        try:
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)
            
            # สำหรับ BytesIO ต้องใช้วิธีพิเศษ
            if isinstance(audio_file, io.BytesIO):
                waveform, sr = torchaudio.load(audio_file, format="wav")
            else:
                waveform, sr = torchaudio.load(audio_file)
            
            if waveform.numel() == 0:
                raise ValueError("ไฟล์เสียงว่างเปล่า")
            
            return waveform.float(), int(sr)
            
        except Exception as e:
            errors.append(f"torchaudio error: {str(e)}")
    
    # ถ้าทั้งสองวิธีล้มเหลว
    error_msg = "ไม่สามารถโหลดไฟล์เสียงได้:\n" + "\n".join(f"- {err}" for err in errors)
    raise Exception(error_msg)

def preprocess_audio(waveform, sr, target_sr=24000):
    """ประมวลผลเสียงก่อนส่งเข้าโมเดล"""
    device = torch.device(CONFIG['device'])
    
    try:
        # Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # ตรวจสอบว่า waveform มีข้อมูล
        if waveform.numel() == 0:
            raise ValueError("ข้อมูลเสียงว่างเปล่า")
        
        # Resample
        if sr != target_sr:
            try:
                # ใช้ torchaudio ถ้ามี
                if TORCHAUDIO_AVAILABLE and T is not None:
                    resampler = T.Resample(sr, target_sr).to(device)
                    waveform = waveform.to(device)
                    waveform = resampler(waveform)
                else:
                    # ใช้ scipy.signal.resample แทน
                    from scipy import signal
                    waveform_np = waveform.squeeze(0).numpy()
                    num_samples = int(len(waveform_np) * target_sr / sr)
                    resampled = signal.resample(waveform_np, num_samples)
                    waveform = torch.from_numpy(resampled).unsqueeze(0).float().to(device)
            except Exception as e:
                raise Exception(f"ไม่สามารถ resample ได้: {str(e)}")
        else:
            waveform = waveform.to(device)
        
        # ตรวจสอบอีกครั้งหลัง resample
        if waveform.numel() == 0:
            raise ValueError("ข้อมูลเสียงหายไปหลัง resample")
        
        return waveform
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการ preprocess: {str(e)}")

def process_audio(audio_file, model):
    """ฟังก์ชันหลักสำหรับแปลงเสียง"""
    try:
        # 1. โหลดไฟล์เสียง
        try:
            waveform, sr = load_audio_file(audio_file)
        except Exception as e:
            raise Exception(f"ขั้นตอนที่ 1 (โหลดไฟล์): {str(e)}")
        
        # 2. Preprocess
        try:
            waveform = preprocess_audio(waveform, sr, CONFIG['sample_rate'])
        except Exception as e:
            raise Exception(f"ขั้นตอนที่ 2 (preprocess): {str(e)}")
        
        # 3. Inference
        try:
            input_tensor = waveform.unsqueeze(0)  # [1, 1, length]
            
            # ตรวจสอบ shape
            if len(input_tensor.shape) != 3:
                raise ValueError(f"Input tensor shape ไม่ถูกต้อง: {input_tensor.shape}")
            
            # ใช้ GPU ถ้ามี
            device = torch.device(CONFIG['device'])
            if device.type == 'cuda':
                # Clear GPU cache ก่อนใช้งาน
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # Clear GPU cache หลังใช้งาน
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if output_tensor.numel() == 0:
                raise ValueError("โมเดลไม่สามารถแปลงเสียงได้ - ผลลัพธ์ว่างเปล่า")
                
        except Exception as e:
            # Clear GPU cache ถ้าเกิด error
            if CONFIG['device'] == 'cuda':
                torch.cuda.empty_cache()
            raise Exception(f"ขั้นตอนที่ 3 (inference): {str(e)}")
        
        # 4. Post-process
        try:
            output_waveform = output_tensor.squeeze(0).cpu()
            
            # ตรวจสอบว่าเป็น tensor ที่ถูกต้อง
            if output_waveform.numel() == 0:
                raise ValueError("ผลลัพธ์ว่างเปล่า")
            
            # Normalize เพื่อป้องกัน clipping
            max_val = torch.abs(output_waveform).max()
            if max_val > 0:
                output_waveform = output_waveform / max_val * 0.95
            
        except Exception as e:
            raise Exception(f"ขั้นตอนที่ 4 (post-process): {str(e)}")
        
        # 5. บันทึกเป็นไฟล์
        try:
            buffer = io.BytesIO()
            
            # ใช้ soundfile ถ้ามี (แนะนำ)
            if SOUNDFILE_AVAILABLE:
                # แปลงเป็น numpy array
                audio_np = output_waveform.squeeze(0).numpy()
                # บันทึกลง buffer
                sf.write(buffer, audio_np, CONFIG['sample_rate'], format='WAV')
                buffer.seek(0)
            # ใช้ torchaudio ถ้าไม่มี soundfile
            elif TORCHAUDIO_AVAILABLE:
                torchaudio.save(
                    buffer,
                    output_waveform,
                    CONFIG['sample_rate'],
                    format="wav"
                )
                buffer.seek(0)
            else:
                raise Exception("ไม่สามารถบันทึกไฟล์ได้ - ต้องมี soundfile หรือ torchaudio")
            
            # ตรวจสอบว่า buffer มีข้อมูล
            if buffer.getvalue() is None or len(buffer.getvalue()) == 0:
                raise ValueError("ไม่สามารถสร้างไฟล์เสียงได้")
            
        except Exception as e:
            raise Exception(f"ขั้นตอนที่ 5 (บันทึกไฟล์): {str(e)}")
        
        return buffer
        
    except Exception as e:
        # เพิ่ม traceback สำหรับ debugging
        error_trace = traceback.format_exc()
        full_error = f"{str(e)}\n\nTraceback:\n{error_trace}"
        raise Exception(full_error)

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
        
        # แสดงข้อมูล Device
        device_name = CONFIG['device'].upper()
        if device_name == "CUDA":
            gpu_info = get_gpu_info()
            st.sidebar.success(f"🚀 GPU: {gpu_info}")
            st.sidebar.info(f"⚙️ Device: {device_name}")
        else:
            st.sidebar.info(f"⚙️ Device: {device_name}")
            if torch.cuda.is_available():
                st.sidebar.warning("⚠️ GPU พบแต่ไม่สามารถใช้งานได้ ใช้ CPU แทน")
            else:
                st.sidebar.info("💻 ใช้ CPU (ไม่มี GPU)")
        
        if SOUNDFILE_AVAILABLE:
            st.sidebar.success("✅ soundfile พร้อมใช้งาน")
        elif TORCHAUDIO_AVAILABLE:
            st.sidebar.warning("⚠️ ใช้ torchaudio (soundfile ไม่พร้อม)")
        else:
            st.sidebar.error("❌ ไม่มี soundfile หรือ torchaudio")
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
            
            # แสดงข้อมูลไฟล์
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.success("✅ อัปโหลดไฟล์สำเร็จ!")
            st.info(f"📄 ไฟล์: {uploaded_file.name} ({file_size:.1f} MB)")
            
            # แสดง audio player (สำหรับไฟล์เล็ก) หรือแสดงข้อความสำหรับไฟล์ใหญ่
            if file_size > 50:
                st.warning(f"⚠️ ไฟล์มีขนาดใหญ่ ({file_size:.1f} MB) - Audio player อาจไม่แสดงได้")
                st.info("💡 ไฟล์จะถูกประมวลผลได้ปกติ แต่การแสดงผลอาจมีปัญหา")
            else:
                try:
                    # Reset file pointer ก่อนแสดง
                    uploaded_file.seek(0)
                    st.audio(uploaded_file, format='audio/wav')
                except Exception as e:
                    try:
                        uploaded_file.seek(0)
                        st.audio(uploaded_file)
                    except:
                        st.warning("⚠️ ไม่สามารถแสดง audio player ได้ แต่ไฟล์พร้อมใช้งาน")
    
    # แสดงปุ่มแปลงเสียง
    if audio_source is not None:
        st.markdown("---")
        st.subheader("2. แปลงเสียง")
        
        # แสดงข้อมูลไฟล์
        try:
            if hasattr(audio_source, 'getvalue'):
                file_size = len(audio_source.getvalue()) / (1024 * 1024)  # MB
                file_info = f" ({file_size:.1f} MB)"
            else:
                file_info = ""
        except:
            file_info = ""
        
        st.caption(f"🎵 เสียงต้นฉบับ: {audio_name}{file_info}")
        
        # แสดง audio player (ถ้าไฟล์ไม่ใหญ่เกินไป)
        try:
            if hasattr(audio_source, 'getvalue'):
                file_size = len(audio_source.getvalue()) / (1024 * 1024)
                if file_size <= 50:
                    if hasattr(audio_source, 'seek'):
                        audio_source.seek(0)
                    try:
                        st.audio(audio_source, format='audio/wav')
                    except:
                        if hasattr(audio_source, 'seek'):
                            audio_source.seek(0)
                        st.audio(audio_source)
                else:
                    st.info(f"📊 ไฟล์ขนาดใหญ่ ({file_size:.1f} MB) - พร้อมแปลงเสียงได้")
            else:
                if hasattr(audio_source, 'seek'):
                    audio_source.seek(0)
                try:
                    st.audio(audio_source, format='audio/wav')
                except:
                    if hasattr(audio_source, 'seek'):
                        audio_source.seek(0)
                    st.audio(audio_source)
        except Exception as e:
            st.info("📊 ไฟล์พร้อมแปลงเสียงได้")
        
        # ปุ่มแปลงเสียง
        if st.button("✨ แปลงเสียงเป็น Anime ✨", type="primary", use_container_width=True):
            error_container = st.container()
            
            try:
                with st.spinner('🤖 AI กำลังร่ายเวทมนตร์... (Converting)'):
                    # Reset file pointer
                    if hasattr(audio_source, 'seek'):
                        try:
                            audio_source.seek(0)
                        except:
                            pass
                    
                    # ตรวจสอบโมเดล
                    if model is None:
                        error_container.error("❌ โมเดลยังไม่พร้อมใช้งาน")
                        st.stop()
                    
                    # แปลงเสียง
                    converted_audio_bytes = process_audio(audio_source, model)
                    
                    if converted_audio_bytes is None:
                        error_container.error("❌ ไม่สามารถแปลงเสียงได้ - ไม่มีผลลัพธ์")
                        st.stop()
                    
                    # แสดงผลลัพธ์
                    st.success("✅ เสร็จเรียบร้อย! (Done)")
                    st.subheader("🎉 ผลลัพธ์ (Anime Voice)")
                    
                    # แสดงเสียง
                    try:
                        st.audio(converted_audio_bytes, format='audio/wav')
                    except Exception as e:
                        st.warning(f"⚠️ ไม่สามารถแสดงเสียงได้: {e}")
                        # ลองอีกครั้งโดยไม่ระบุ format
                        try:
                            converted_audio_bytes.seek(0)
                            st.audio(converted_audio_bytes)
                        except:
                            pass
                    
                    # ปุ่มดาวน์โหลด
                    try:
                        converted_audio_bytes.seek(0)
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
                error_container.error("❌ เกิดข้อผิดพลาดขณะแปลงเสียง")
                error_container.error(f"**รายละเอียด**: {str(e)}")
                
                # แสดงรายละเอียด error
                with st.expander("🔍 ดูรายละเอียดข้อผิดพลาด (สำหรับ Developer)"):
                    st.code(traceback.format_exc(), language="python")
                
                # คำแนะนำ
                st.info("💡 **คำแนะนำ**:")
                st.info("1. ตรวจสอบว่าไฟล์เสียงไม่เสียหาย")
                st.info("2. ลองใช้ไฟล์เสียงอื่น (WAV format แนะนำ)")
                st.info("3. ตรวจสอบว่าไฟล์เสียงมีขนาดไม่ใหญ่เกินไป (< 10MB)")
                st.info("4. ลองรีเฟรชหน้าเว็บและลองอีกครั้ง")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by PyTorch & Streamlit | Model: Simple Conv1d RVC")

# Run the app
if __name__ == "__main__":
    main()
