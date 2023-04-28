from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import streamlit as st
import os

ALLOWED_EXTENSIONS = {'wav'}

model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file):
    est_sources = model.separate_file(path=file) 
    return est_sources[:, :, 0].detach().cpu()

def main():
    st.set_page_config(page_title="Speech Enhancement", page_icon="🔊", layout="wide")
    st.title("Speech Enhancement")

    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            with st.spinner("Processing..."):
                enhanced = process_file(uploaded_file)
            speech, sr = torchaudio.load(uploaded_file)
            st.audio(speech, format='audio/wav', start_time=0, caption="Original Audio",  sample_rate=sr)
            st.audio(enhanced, format='audio/wav', start_time=0, caption="Enhanced Audio",  sample_rate=8000)
        else:
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == '__main__':
    main()
