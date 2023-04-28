import streamlit as st
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio


def run_audio_enhancement():
    # Load the model
    model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')

    # Create a file uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Load the audio file
        audio_data = uploaded_file.read()
        audio_tensor, sample_rate = torchaudio.load(audio_data)

        # Apply the SepFormer model to the audio file
        est_sources = model.separate_tensor(audio_tensor)

        # Save the enhanced audio file
        torchaudio.save("enhanced_audio.wav", est_sources[:, :, 0].detach().cpu(), sample_rate)

        # Display the original and enhanced audio files
        st.audio(audio_data, format='audio/wav')
        st.audio("enhanced_audio.wav", format='audio/wav')


def main():
    # Set the title and description
    st.set_page_config(page_title="Audio Enhancement with SepFormer")
    st.title("Audio Enhancement with SepFormer")
    st.write("This app uses the SepFormer model to enhance audio.")

    # Run the audio enhancement function
    run_audio_enhancement()


if __name__ == "__main__":
    main()
