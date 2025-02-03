from clearvoice import ClearVoice
import argparse

def process_audio(input_path, output_path="output_directory"):
    """
    Process audio files in the given input directory and save the separated outputs.
    
    :param input_path: Path to the directory containing input WAV files.
    :param output_path: Path to the directory where separated audio will be saved.
    """
    myClearVoice = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    
    # Get output dictionary with separated audio
    output_wav_dict = myClearVoice(
        input_path=input_path, online_write=True, output_path=output_path
    )
    
    return output_wav_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Separation using ClearVoice")
    parser.add_argument("--input_path", type=str, help="Path to the input directory containing WAV files")
    parser.add_argument("--output_path", type=str, default="output_directory", help="Path to save separated audio")
    
    args = parser.parse_args()
    
    output = process_audio(args.input_path, args.output_path)
    print("Processing complete. Output saved to:", args.output_path)