import os
import time
import torch
import torchaudio
import sys
from multiprocessing import Pool, cpu_count
import pandas as pd
import opensmile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.IemocapDataset import IemocapDataset

def process_segment(segment, emotion, filename, output_file_name, smile, specgram_transform):
    try:
        # Ensure the segment is 2D by adding the channel dimension
        segment_2d = segment.unsqueeze(0)  # Shape: [1, num_frames]

        # Unique filename for each process
        segment_filename = f'segment_{os.getpid()}.wav'
        torchaudio.save(segment_filename, segment_2d, sample_rate=16000)

        # Extract emotion features using OpenSMILE
        features = smile.process_file(segment_filename)

        # Append the features to the output CSV
        if not os.path.exists(output_file_name):
            features.to_csv(output_file_name, index=False)
        else:
            features.to_csv(output_file_name, mode='a', header=False, index=False)

        # Extract spectrograms using torchaudio transform
        specgram = specgram_transform(segment_2d)
        
        os.remove(segment_filename)  # Remove the temporary .wav file
        return specgram  # Return the spectrogram if needed for further processing

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    # Log start time
    start_time = time.time()

    # Specify file path and name (include file extension)
    output_file_name = 'iemocap_GeMAPSv01b_emotion.csv'

    # Load dataset and construct dataloader
    dataset = IemocapDataset(r"M:\SOL\Sentiment_detection\IEMOCAP\IEMOCAP_full_release")
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=16, 
                                             shuffle=False, 
                                             collate_fn=IemocapDataset.collate_fn_segments)

    # Remove output file if exists
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    # Initialize OpenSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Initialize stft spectrogram transform
    specgram_transform = torchaudio.transforms.Spectrogram(n_fft=256, 
                                                           win_length=256, 
                                                           hop_length=128)
    
    # # Initialize MEL spectrogram transform
    # specgram_transform = torchaudio.transforms.MelSpectrogram(n_fft=512, win_length=512, hop_length=256, n_mels=80)

    # Iterate through dataset using dataloader to get segments of each utterance
    for segments, emotions, n_segments, filenames in dataloader:
        with Pool(cpu_count()) as pool:pool.starmap(process_segment, [(segment, emotion, filename, output_file_name, smile, specgram_transform) for segment, emotion, filename in zip(segments, emotions, filenames)])

    # Compute and print program execution time
    end_time = time.time()
    total_time = end_time - start_time
    print('Program took %d min %d sec to complete' % (total_time // 60, total_time % 60))

if __name__ == '__main__':
    main()
