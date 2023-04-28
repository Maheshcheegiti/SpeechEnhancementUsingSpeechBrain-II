import streamlit as st
import torch
import torchaudio
from speechbrain.pretrained import SepformerSeparation as Separator

def main():
    st.title("Speech Enhancement using Sepformer Model")

    # Load the pretrained model
    model = Separator.from_hparams(
        source="speechbrain/sepformer-whamr-enhancement",
        savedir='pretrained_models/sepformer-whamr-enhancement'
    )

    # Create a file uploader for the user to upload an audio file
    file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    # If the user has uploaded a file
    if file is not None:
        # Load the audio file into a Tensor
        audio, sample_rate = torchaudio.load(file)

        # Perform speech enhancement on the audio
        est_sources = model.separate(
            audio,
            sample_rate=sample_rate,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Save the enhanced audio to a file
        torchaudio.save("enhanced_audio.wav", est_sources[:, :, 0].detach().cpu(), sample_rate)

        # Display the enhanced audio to the user
        st.audio("enhanced_audio.wav")

if __name__ == "__main__":
    main()
