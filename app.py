import streamlit as st
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

def main():
    st.title("Speech Enhancement using SpeechBrain - SepFormer")
    st.write("This app enhances the speech in an audio file using the SpeechBrain Sepformer model.")

    file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
    if file:
        audio_bytes = file.read()
        st.text("Original Audio")
        st.audio(audio_bytes, format="audio/wav")

        with open("uploaded_file.wav", "wb") as f:
            f.write(audio_bytes)

        model = separator.from_hparams(
            source="speechbrain/sepformer-whamr-enhancement",
            savedir="pretrained_models/sepformer-whamr-enhancement",
        )

        est_sources = model.separate_file(path="uploaded_file.wav")

        torchaudio.save("enhanced_audio.wav", est_sources[:, :, 0].detach().cpu(), 8000)

        with open("enhanced_audio.wav", "rb") as f:
            audio_bytes = f.read()
        st.text("Enhanced Audio")
        st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    main()
