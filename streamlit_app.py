import streamlit as st
import librosa
import soundfile as sf
import torch

# โหลดโมเดล RVC
MODEL_PATH = "model.pth"
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

# ฟังก์ชันแปลงเสียง
def convert_audio(input_audio_path):
    waveform, sample_rate = librosa.load(input_audio_path, sr=16000)
    
    # เรียกใช้โมเดล RVC
    with torch.no_grad():
        output_waveform = model(waveform)

    output_audio_path = "converted_audio.wav"
    sf.write(output_audio_path, output_waveform.squeeze().numpy(), sample_rate)

    return output_audio_path

# UI ด้วย Streamlit
st.title("Anime Voice Converter (RVC)")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    input_audio_path = f"temp_{uploaded_file.name}"
    with open(input_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    output_audio_path = convert_audio(input_audio_path)
    
    st.audio(output_audio_path, format="audio/wav")
    st.download_button("Download Converted Audio", output_audio_path)
