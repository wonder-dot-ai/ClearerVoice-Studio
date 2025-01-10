---
license: apache-2.0
---
# Introduction

The MossFormer2_SR_48K model weights for 48 kHz speech super-resolution in [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio/tree/main) repo.

This model is trained on large scale datasets inclduing open-sourced and private data.

The purpose is to enhance the quality of speech signals by increasing their temporal and spectral resolution, typically by converting low-resolution (low sampling rate) 
audio to high-resolution (high sampling rate) audio. This involves reconstructing the high-frequency components that are often missing in low-resolution signals.

# Install

**Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

**Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n clearvoice python=3.12.1
conda activate clearvoice
pip install -r requirements.txt
```

**Run Script**

Go to `clearvoice/` and use the following examples. The MossFormer2_SR_48K model will be downloaded from huggingface automatically.

Sample example 1: use model `MossFormer2_SR_48K` to process one wave file of `samples/input.wav` and save the output wave file to `samples/output_MossFormer2_SR_48K.wav`

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)

myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SR_48K.wav')
```

Sample example 2: use speech enhancement model `MossFormer2_SE_48K` to process all input wave files in `samples/path_to_input_wavs/` and save all output files to `samples/path_to_output_wavs`

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')
```

Sample example 3: use speech enhancement model `MossFormer2_SE_48K` to process wave files listed in `samples/audio_samples.scp' file, and save all output files to 'samples/path_to_output_wavs_scp/'

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
```

Model Limitations: The current speech super-resolution model is trained on a clean speech dataset and is designed to work with clean speech inputs. For speech super-resolution on noisy speech audio, 
we recommend using our 'MossFormer2_SE_48K' model for speech enhancement first, followed by 'MossFormer2_SR_48K' for speech super-resolution.