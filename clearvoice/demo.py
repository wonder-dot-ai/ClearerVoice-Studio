from clearvoice import ClearVoice

##-----Demo Six: use MossFormer2_SS_16K model for speech separation -----------------
myClearVoice = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])

# Process all wav files in input directory and save outputs to output directory
input_path = "input_directory"
output_path = "output_directory"

# Get output dictionary with separated audio
output_wav_dict = myClearVoice(
    input_path=input_path, online_write=True, output_path=output_path
)