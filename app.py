import gradio as gr
import torch
import faiss
import numpy as np
import librosa
import soundfile as sf
from some_rvc_module import load_model, process_audio  # เปลี่ยนเป็นโมดูลที่ใช้จริง

# โหลดโมเดล RVC
model_path = "model/model.pth"
index_path = "model/model.index"

# โหลดโมเดล
model = load_model(model_path)
index = faiss.read_index(index_path)

def voice_conversion(input_audio):
    # โหลดไฟล์เสียง
    waveform, sr = librosa.load(input_audio, sr=44100)
    
    # ประมวลผลเสียงด้วยโมเดล RVC
    anime_voice = process_audio(waveform, model, index)
    
    # บันทึกไฟล์เสียงที่แปลงแล้ว
    output_path = "anime_voice.wav"
    sf.write(output_path, anime_voice, sr)
    
    return output_path, output_path  # คืนค่าไฟล์สำหรับเล่นและดาวน์โหลด

# สร้าง UI ด้วย Gradio
interface = gr.Interface(
    fn=voice_conversion,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=[gr.Audio(type="filepath"), gr.File()],
    title="Anime Voice Converter",
    description="อัปโหลดไฟล์เสียง แล้วเปลี่ยนเป็นเสียงอนิเมะ!",
)

interface.launch()
