import streamlit as st
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(
    source="speechbrain/sepformer-whamr-enhancement",
    savedir='pretrained_models/sepformer-whamr-enhancement'
)

def process_file(path):
    est_sources = model.separate_file(path=path)
    return est_sources

def main():
    st.set_page_config(page_title="Speech Enhancement", page_icon="🔊", layout="wide")
    st.title("Speech Enhancement - SpeechBrain - SepFormer")

    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Save uploaded file to disk
            with open("uploaded_file.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process file
            est_sources = process_file(path="uploaded_file.wav")

            # Save enhanced signal to disk
            torchaudio.save("enhanced.wav", est_sources[:, :, 0].detach().cpu(), 8000)

            # Load original audio
            original_audio, sr = torchaudio.load("uploaded_file.wav")

            # Show original and enhanced signals to user
            st.text("Original Audio")
            st.audio(original_audio.numpy(), format='audio/wav', start_time=0, caption="Original Audio")
            st.text("Enhanced Audio")
            st.audio("enhanced.wav", format='audio/wav', start_time=0, caption="Enhanced Audio")

if __name__ == '__main__':
    main()
