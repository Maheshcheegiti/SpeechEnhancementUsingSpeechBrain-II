import streamlit as st
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(
    source="speechbrain/sepformer-whamr-enhancement",
    savedir="pretrained_models/sepformer-whamr-enhancement",
)

ALLOWED_EXTENSIONS = {"wav"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file):
    speech, rate = torchaudio.load(file)
    assert rate == 8000, "mismatch in sampling rate"
    est_sources = model.separate_waveform(speech[0])
    return speech[0], est_sources[0][0], rate

def main():
    st.set_page_config(page_title="Speech Enhancement", page_icon="🔊", layout="wide")

    st.title("Speech Enhancement - SpeechBrain - SepFormer")

    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            with st.spinner("Processing..."):
                speech, enhanced, sr = process_file(uploaded_file)
            st.audio(speech, format="audio/wav", start_time=0, sample_rate=sr)
            st.text("Original audio:")
            st.audio(enhanced, format="audio/wav", start_time=0, sample_rate=sr)
            st.text("Enhanced audio:")
        else:
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == "__main__":
    main()
