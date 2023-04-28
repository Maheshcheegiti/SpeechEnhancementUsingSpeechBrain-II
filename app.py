import streamlit as st
import torchaudio
from speechbrain.pretrained import SepformerSeparation as separator

model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')

def process_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    est_sources = model.separate_file(path=file_path)
    return waveform, est_sources[:, :, 0], sample_rate

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'wav'

def main():
    st.set_page_config(page_title="Speech Enhancement", page_icon="🔊", layout="wide")

    st.title("Speech Enhancement - SpeechBrain - SepFormer")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
        with st.spinner("Processing..."):
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            speech, enhanced, sr = process_file(file_path)
        st.audio(speech, format='audio/wav', start_time=0, caption="Original Audio",  sample_rate=sr)
        st.audio(enhanced, format='audio/wav', start_time=0, caption="Enhanced Audio", sample_rate=sr)
    else:
        st.warning("Invalid file type. Please upload a WAV file.")
if __name__ == '__main__':
    main()
