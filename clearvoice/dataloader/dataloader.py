import numpy as np
import math, os, csv
import torchaudio
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import soundfile as sf
from torch.utils.data import Dataset
import torch.utils.data as data
import os 
import sys
sys.path.append(os.path.dirname(__file__))
from pydub import AudioSegment
from dataloader.misc import read_and_config_file, get_file_extension
from datasets import load_dataset
from pathlib import Path
import librosa
import random
EPS = 1e-6
MAX_WAV_VALUE_16B = 32768.0
MAX_WAV_VALUE_32B = 2147483648.0
   
def audioread_archieved(path, sampling_rate):
    """
    Reads an audio file from the specified path, normalizes the audio, 
    resamples it to the desired sampling rate (if necessary), and ensures it is single-channel.

    Parameters:
    path (str): The file path of the audio file to be read.
    sampling_rate (int): The target sampling rate for the audio.

    Returns:
    numpy.ndarray: The processed audio data, normalized, resampled (if necessary),
                   and converted to mono (if the input audio has multiple channels).
    """
    
    # Read audio data and its sample rate from the file.
    data, fs = sf.read(path)

    # convert to mono channel
    if len(data.shape) >1:
        if data.shape[0] > data.shape[1]:
            data = data[:, 0] 
        else:
            data = data[0, :] 
    
    # Normalize the audio data.
    data, scalar = audio_norm(data)
    
    # Resample the audio if the sample rate is different from the target sampling rate.
    if fs != sampling_rate:
        data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
    
    # Convert to mono by selecting the first channel if the audio has multiple channels.
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Return the processed audio data.
    return data, scalar
    
def read_audio(file_path):
    """
    Use AudioSegment to load audio from all supported audio input format
    """
    
    try:
        audio = AudioSegment.from_file(file_path)
        return audio
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
        
def audioread(path, sampling_rate, use_norm):
    """
    Reads an audio file from the specified path, normalizes the audio,
    resamples it to the desired sampling rate (if necessary), and ensures it is single-channel.

    Parameters:
    path (str): The file path of the audio file to be read.
    sampling_rate (int): The target sampling rate for the audio.
    use_norm (bool): The flag for specifying whether using input audio normalization

    Returns:
    numpy.ndarray: The processed audio data, normalized, resampled (if necessary),
                   and converted to mono (if the input audio has multiple channels).
    """
    
    # Read audio data and its sample rate from the file.
    audio_info = {}
    ext = get_file_extension(path).replace('.', '')
    audio_info['ext']=ext
    
    try:
        data = AudioSegment.from_file(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
        
    data = read_audio(path)
   
    audio_info['sample_rate'] = data.frame_rate
    audio_info['channels'] = data.channels
    audio_info['sample_width'] = data.sample_width

    data_array = np.array(data.get_array_of_samples())
    if max(data_array) > MAX_WAV_VALUE_16B:
        audio_np = data_array / MAX_WAV_VALUE_32B
    else:
        audio_np = data_array / MAX_WAV_VALUE_16B

    audios = []
    # Check if the audio is stereo
    if audio_info['channels'] == 2:
        audios.append(audio_np[::2])  # Even indices (left channel)
        audios.append(audio_np[1::2])  # Odd indices (right channel)
    else:
        audios.append(audio_np)
    
    # Normalize the audio data.
    audios_normed = []
    scalars = []
    for audio in audios:
        if use_norm:
            audio_normed, scalar = audio_norm(audio)
            audios_normed.append(audio_normed)
            scalars.append(scalar)
        else:
            audios_normed.append(audio)
            scalars.append(1)
    # Resample the audio if the sample rate is different from the target sampling rate.
    if audio_info['sample_rate'] != sampling_rate:
        index = 0
        for audio_normed in audios_normed:
            audios_normed[index] = librosa.resample(audio_normed, orig_sr=audio_info['sample_rate'], target_sr=sampling_rate)
            index = index + 1
    
    # Return the processed audio data.
    return audios_normed, scalars, audio_info

def audio_norm(x):
    """
    Normalizes the input audio signal to a target Root Mean Square (RMS) level, 
    applying two stages of scaling. This ensures the audio signal is neither too quiet 
    nor too loud, keeping its amplitude consistent.

    Parameters:
    x (numpy.ndarray): Input audio signal to be normalized.

    Returns:
    numpy.ndarray: Normalized audio signal.
    """
    
    # Compute the root mean square (RMS) of the input audio signal.
    rms = (x ** 2).mean() ** 0.5
    
    # Calculate the scalar to adjust the signal to the target level (-25 dB).
    scalar = 10 ** (-25 / 20) / (rms + EPS)
    
    # Scale the input audio by the computed scalar.
    x = x * scalar
    
    # Compute the power of the scaled audio signal.
    pow_x = x ** 2
    
    # Calculate the average power of the audio signal.
    avg_pow_x = pow_x.mean()
    
    # Compute RMS only for audio segments with higher-than-average power.
    rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
    
    # Calculate another scalar to further normalize based on higher-power segments.
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)
    
    # Apply the second scalar to the audio.
    x = x * scalarx
    
    # Return the doubly normalized audio signal.
    return x, 1/(scalar * scalarx + EPS)

class DataReader:
    def __init__(self, args):
        self.args = args
        self.sampling_rate = args.sampling_rate
        self.dataset_name = args.input_path
        self.split = "train"
        try:
            self.dataset = load_dataset(
                self.dataset_name, split=self.split, streaming=True
            )
        except Exception as e:
            raise ValueError(
                f"Error loading Hugging Face Dataset: {e}. Ensure '{self.dataset_name}' is correct."
            )

    def __iter__(self):
        for index, item in enumerate(self.dataset):
            yield self.extract_feature(item, index)
        
    def __len__(self):
        raise TypeError("IterableDataset doesn't have length")

    def __getitem__(self, index):
        raise TypeError("IterableDataset doesn't support indexing")
        

    def extract_feature(self, item, index):
        use_norm = self.args.network in ["FRCRN_SE_16K", "MossFormer2_SS_16K"]

        try:
            utt_id = f"{Path(item['audio']['path']).stem}.wav"
        except (KeyError, TypeError):
            utt_id = f"sample_{index}"

        audio_array = item["audio"]["array"]
        sampling_rate_hf = item["audio"]["sampling_rate"]
        audios_norm, scalars, audio_info = self.process_hf_audio_array(
            audio_array, sampling_rate_hf, self.sampling_rate, use_norm
        )

        if self.args.network == "MossFormer2_SR_48K":
            audio_info["sample_rate"] = self.sampling_rate

        for i in range(len(audios_norm)):
            audios_norm[i] = audios_norm[i].astype(np.float32)
            audios_norm[i] = np.reshape(audios_norm[i], [1, audios_norm[i].shape[0]])

        return audios_norm, utt_id, audios_norm[0].shape[1], scalars, audio_info

    def process_hf_audio_array(self, audio_array, original_sr, target_sr, use_norm):
        audio_np = np.array(audio_array)
        audios = [audio_np]

        audios_normed, scalars = [], []
        for audio in audios:
            if use_norm:
                audio_normed, scalar = audio_norm(audio)
                audios_normed.append(audio_normed)
                scalars.append(scalar)
            else:
                audios_normed.append(audio)
                scalars.append(1)

        if original_sr != target_sr:
            for i in range(len(audios_normed)):
                audios_normed[i] = librosa.resample(
                    audios_normed[i], orig_sr=original_sr, target_sr=target_sr
                )

        audio_info = {
            "sample_rate": target_sr,
            "channels": 1,
            "sample_width": 2,
            "ext": "wav",
        }
        return audios_normed, scalars, audio_info

class Wave_Processor(object):
    """
    A class for processing audio data, specifically for reading input and label audio files,
    segmenting them into fixed-length segments, and applying padding or trimming as necessary.

    Methods:
    process(path, segment_length, sampling_rate):
        Processes audio data by reading, padding, or segmenting it to match the specified segment length.
    
    Parameters:
    path (dict): A dictionary containing file paths for 'inputs' and 'labels' audio files.
    segment_length (int): The desired length of audio segments to extract.
    sampling_rate (int): The target sampling rate for reading the audio files.
    """

    def process(self, path, segment_length, sampling_rate):
        """
        Reads input and label audio files, and ensures the audio is segmented into
        the desired length, padding if necessary or extracting random segments if
        the audio is longer than the target segment length.

        Parameters:
        path (dict): Dictionary containing the paths to 'inputs' and 'labels' audio files.
        segment_length (int): Desired length of the audio segment in samples.
        sampling_rate (int): Target sample rate for the audio.

        Returns:
        tuple: A pair of numpy arrays representing the processed input and label audio,
               either padded to the segment length or trimmed.
        """
        # Read the input and label audio files using the target sampling rate.
        wave_inputs = audioread(path['inputs'], sampling_rate)
        wave_labels = audioread(path['labels'], sampling_rate)
        
        # Get the length of the label audio (assumed both inputs and labels have similar lengths).
        len_wav = wave_labels.shape[0]
        
        # If the input audio is shorter than the desired segment length, pad it with zeros.
        if wave_inputs.shape[0] < segment_length:
            # Create zero-padded arrays for inputs and labels.
            padded_inputs = np.zeros(segment_length, dtype=np.float32)
            padded_labels = np.zeros(segment_length, dtype=np.float32)
            
            # Copy the original audio into the padded arrays.
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_labels[:wave_labels.shape[0]] = wave_labels
        else:
            # Randomly select a start index for segmenting the audio if it's longer than the segment length.
            st_idx = random.randint(0, len_wav - segment_length)
            
            # Extract a segment of the desired length from the inputs and labels.
            padded_inputs = wave_inputs[st_idx:st_idx + segment_length]
            padded_labels = wave_labels[st_idx:st_idx + segment_length]
        
        # Return the processed (padded or segmented) input and label audio.
        return padded_inputs, padded_labels

class Fbank_Processor(object):
    """
    A class for processing input audio data into mel-filterbank (Fbank) features, 
    including the computation of delta and delta-delta features.
    
    Methods:
    process(inputs, args):
        Processes the raw audio input and returns the mel-filterbank features 
        along with delta and delta-delta features.
    """
    
    def process(self, inputs, args):
        # Convert frame length and shift from seconds to milliseconds.
        frame_length = int(args.win_len / args.sampling_rate * 1000)
        frame_shift = int(args.win_inc / args.sampling_rate * 1000)

        # Set up configuration for the mel-filterbank computation.
        fbank_config = {
            "dither": 1.0,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "num_mel_bins": args.num_mels,
            "sample_frequency": args.sampling_rate,
            "window_type": args.win_type
        }

        # Convert the input audio to a FloatTensor and scale it to match the expected input range.
        inputs = torch.FloatTensor(inputs * MAX_WAV_VALUE)

        # Compute the mel-filterbank features using Kaldi's fbank function.
        fbank = torchaudio.compliance.kaldi.fbank(inputs.unsqueeze(0), **fbank_config)

        # Add delta and delta-delta features.
        fbank_tr = torch.transpose(fbank, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        
        # Concatenate the original Fbank, delta, and delta-delta features.
        fbanks = torch.cat([fbank, fbank_delta, fbank_delta_delta], dim=1)
        
        return fbanks.numpy()

class AudioDataset(Dataset):
    """
    A dataset class for loading and processing audio data from different data types 
    (train, validation, test). Supports audio processing and feature extraction 
    (e.g., waveform processing, Fbank feature extraction).

    Parameters:
    args: Arguments containing dataset configuration (paths, sampling rate, etc.).
    data_type (str): The type of data to load (train, val, test).
    """

    def __init__(self, args, data_type):
        self.args = args
        self.sampling_rate = args.sampling_rate
        
        # Read the list of audio files based on the data type.
        if data_type == 'train':
            self.wav_list = read_and_config_file(args.tr_list)
        elif data_type == 'val':
            self.wav_list = read_and_config_file(args.cv_list)
        elif data_type == 'test':
            self.wav_list = read_and_config_file(args.tt_list)
        else:
            print(f'Data type: {data_type} is unknown!')
        
        # Initialize processors for waveform and Fbank features.
        self.wav_processor = Wave_Processor()
        self.fbank_processor = Fbank_Processor()
        
        # Clip data to a fixed segment length based on the sampling rate and max length.
        self.segment_length = self.sampling_rate * self.args.max_length
        print(f'No. {data_type} files: {len(self.wav_list)}')

    def __len__(self):
        # Return the number of audio files in the dataset.
        return len(self.wav_list)

    def __getitem__(self, index):
        # Get the input and label paths from the list.
        data_info = self.wav_list[index]
        
        # Process the waveform inputs and labels.
        inputs, labels = self.wav_processor.process(
            {'inputs': data_info['inputs'], 'labels': data_info['labels']}, 
            self.segment_length, 
            self.sampling_rate
        )
        
        # Optionally load Fbank features if specified.
        if self.args.load_fbank is not None:
            fbanks = self.fbank_processor.process(inputs, self.args)
            return inputs * MAX_WAV_VALUE, labels * MAX_WAV_VALUE, fbanks
        
        return inputs, labels

def zero_pad_concat(self, inputs):
    """
    Concatenates a list of input arrays, applying zero-padding as needed to ensure 
    they all match the length of the longest input.

    Parameters:
    inputs (list of numpy arrays): List of input arrays to be concatenated.

    Returns:
    numpy.ndarray: A zero-padded array with concatenated inputs.
    """
    
    # Get the maximum length among all inputs.
    max_t = max(inp.shape[0] for inp in inputs)
    
    # Determine the shape of the output based on the input dimensions.
    shape = None
    if len(inputs[0].shape) == 1:
        shape = (len(inputs), max_t)
    elif len(inputs[0].shape) == 2:
        shape = (len(inputs), max_t, inputs[0].shape[1])
    
    # Initialize an array with zeros to hold the concatenated inputs.
    input_mat = np.zeros(shape, dtype=np.float32)
    
    # Copy the input data into the zero-padded array.
    for e, inp in enumerate(inputs):
        if len(inp.shape) == 1:
            input_mat[e, :inp.shape[0]] = inp
        elif len(inp.shape) == 2:
            input_mat[e, :inp.shape[0], :] = inp
    
    return input_mat

def collate_fn_2x_wavs(data):
    """
    A custom collate function for combining batches of waveform input and label pairs.

    Parameters:
    data (list): List of tuples (inputs, labels).

    Returns:
    tuple: Batched inputs and labels as torch.FloatTensors.
    """
    inputs, labels = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    return x, y

def collate_fn_2x_wavs_fbank(data):
    """
    A custom collate function for combining batches of waveform inputs, labels, and Fbank features.

    Parameters:
    data (list): List of tuples (inputs, labels, fbanks).

    Returns:
    tuple: Batched inputs, labels, and Fbank features as torch.FloatTensors.
    """
    inputs, labels, fbanks = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    z = torch.FloatTensor(fbanks)
    return x, y, z

class DistributedSampler(data.Sampler):
    """
    Sampler for distributed training. Divides the dataset among multiple replicas (processes), 
    ensuring that each process gets a unique subset of the data. It also supports shuffling 
    and managing epochs.

    Parameters:
    dataset (Dataset): The dataset to sample from.
    num_replicas (int): Number of processes participating in the training.
    rank (int): Rank of the current process.
    shuffle (bool): Whether to shuffle the data or not.
    seed (int): Random seed for reproducibility.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Shuffle the indices based on the epoch and seed.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ind = torch.randperm(int(len(self.dataset) / self.num_replicas), generator=g) * self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind + i).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Add extra samples to make the dataset evenly divisible.
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample for the current process.
        indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, data_type):
    """
    Creates and returns a data loader and sampler for the specified dataset type (train, validation, or test).
    
    Parameters:
    args (Namespace): Configuration arguments containing details such as batch size, sampling rate, 
                      network type, and whether distributed training is used.
    data_type (str): The type of dataset to load ('train', 'val', 'test').
    
    Returns:
    sampler (DistributedSampler or None): The sampler for distributed training, or None if not used.
    generator (DataLoader): The PyTorch DataLoader for the specified dataset.
    """
    
    # Initialize the dataset based on the given arguments and dataset type (train, val, or test).
    datasets = AudioDataset(args=args, data_type=data_type)

    # Create a distributed sampler if distributed training is enabled; otherwise, use no sampler.
    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,  # Number of replicas in distributed training.
        rank=args.local_rank  # Rank of the current process.
    ) if args.distributed else None

    # Select the appropriate collate function based on the network type.
    if args.network == 'FRCRN_SE_16K' or args.network == 'MossFormerGAN_SE_16K':
        # Use the collate function for two-channel waveform data (inputs and labels).
        collate_fn = collate_fn_2x_wavs
    elif args.network == 'MossFormer2_SE_48K':
        # Use the collate function for waveforms along with Fbank features.
        collate_fn = collate_fn_2x_wavs_fbank
    else:
        # Print an error message if the network type is unknown.
        print(f'in dataloader, please specify a correct network type using args.network!')
        return

    # Create a DataLoader with the specified dataset, batch size, and worker configuration.
    generator = data.DataLoader(
        datasets,
        batch_size=args.batch_size,  # Batch size for training.
        shuffle=(sampler is None),  # Shuffle the data only if no sampler is used.
        collate_fn=collate_fn,  # Use the selected collate function for batching data.
        num_workers=args.num_workers,  # Number of workers for data loading.
        sampler=sampler  # Use the distributed sampler if applicable.
    )
    
    # Return both the sampler and DataLoader (generator).
    return sampler, generator