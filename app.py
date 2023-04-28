import streamlit as st
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import os

# Download the model and set the save directory
model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')

# Define a function to process the uploaded file
def process_file(uploaded_file):
    # Save the uploaded file to the disk
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = os.path.join("uploads", uploaded_file.name)

    # Separate the audio sources using the model
    est_sources = model.separate_file(path=file_path)

    # Save the enhanced audio to the disk
    torchaudio.save(os.path.join("uploads", "enhanced_" + uploaded_file.name), est_sources[:, :, 0].detach().cpu(), 8000)

    return file_path, os.path.join("uploads", "enhanced_" + uploaded_file.name)

# Define the main function
def main():
    st.title("Speech Enhancement using SpeechBrain")
    st.write("Upload a noisy audio file to enhance it.")

    # Create an uploader widget
    uploaded_file = st.file_uploader("Choose a file")

    # If a file is uploaded
    if uploaded_file is not None:
        # Process the uploaded file
        file_path, enhanced_file_path = process_file(uploaded_file)

        # Show the original audio
        st.audio(file_path, format="audio/wav", caption="Original Audio")

        # Show the enhanced audio
        st.audio(enhanced_file_path, format="audio/wav", caption="Enhanced Audio")

# Run the app
if __name__ == "__main__":
    main()
