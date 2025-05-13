import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import shutil
from pydub.silence import detect_nonsilent

def audio_dataset(url_data, url_mp3, url_wav, drive_output_folder):
    """
    Processes an audio dataset by:
    - Loading filenames and transcriptions
    - Matching .mp3 files
    - Converting them to .wav
    - Cleaning up unrelated files
    - Saving metadata and sorted lists to Drive
    - Returning sorted list of .wav files
    
    Parameters:
    - url_data: Path to the .tsv file with 'path' and 'sentence'
    - url_mp3: Path to folder with source .mp3 files
    - url_wav: Path to save converted .wav files
    - drive_output_folder: Folder in Google Drive to save outputs
    """

    # Step 1: Load and filter data
    data = pd.read_csv(url_data, sep="\t")
    data = data[['path', 'sentence']]
    data_list = data['path'].to_list()
    sorted_list = sorted(data_list)

    # Save filtered DataFrame and sorted list to Google Drive
    os.makedirs(drive_output_folder, exist_ok=True)
    data.to_csv(os.path.join(drive_output_folder, 'filtered_data.csv'), index=False)
    pd.DataFrame(sorted_list, columns=["sorted_paths"]).to_csv(os.path.join(drive_output_folder, 'sorted_list.csv'), index=False)
    print("📁 Saved filtered data and sorted list to Drive.")

    # Step 2: Prepare WAV folder
    os.makedirs(url_wav, exist_ok=True)
    wanted_files = [file for file in sorted_list if file.endswith('.mp3')]
    matched_files = [f for f in os.listdir(url_mp3) if f in wanted_files]

    # Step 3: Convert MP3 to WAV with progress bar
    print("🎧 Converting MP3 to WAV...")
    for file in tqdm(matched_files):
        mp3_path = os.path.join(url_mp3, file)
        wav_path = os.path.join(url_wav, file.replace(".mp3", ".wav"))
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

    # Step 4: Cleanup unrelated .wav files
    allowed_files = {name.replace('.mp3', '.wav') for name in sorted_list}
    all_files = os.listdir(url_wav)

    for fname in all_files:
        if fname.endswith('.wav') and fname not in allowed_files:
            fpath = os.path.join(url_wav, fname)
            os.remove(fpath)
            print(f"🗑️ Deleted: {fname}")

    print("✅ Cleanup complete. Only expected .wav files are kept.")

    # Step 5: Return sorted list of existing .wav files
    ordered_wav_files = [name.replace('.mp3', '.wav') for name in sorted_list]
    existing_wavs = set(os.listdir(url_wav))
    sorted_wav_files = [f for f in ordered_wav_files if f in existing_wavs]

    print("✅ Final sorted list ready.")
    return sorted_wav_files




def preprocess_language_dataset(
    transcript_path,
    wav_source_dir,
    wav_target_dir,
    sample_rate=16000,
    min_duration_ms=500,
    max_duration_ms=15000,
    file_extension='.wav'
):
    """
    Preprocesses .wav files based on transcript order:
    - Converts sample rate, mono channel
    - Trims silence
    - Normalizes loudness
    - Filters by duration
    - Saves processed files in new folder
    
    Parameters:
        transcript_path (str): Path to .tsv or .csv file with 'path' and 'sentence'.
        wav_source_dir (str): Directory with original .wav files.
        wav_target_dir (str): Directory to save processed .wav files.
        sample_rate (int): Target audio sample rate (e.g., 16000).
        min_duration_ms (int): Minimum duration of audio (in ms).
        max_duration_ms (int): Maximum duration of audio (in ms).
        file_extension (str): Audio file extension, default '.wav'.

    Returns:
        missing_list (list): Audio names from sorted list not found in source folder.
        processed_list (list): Processed audio filenames (in order).
    """
    
    # Load and sort transcript data
    df = pd.read_csv(transcript_path, sep='\t')
    df = df[['path', 'sentence']]
    sorted_list = sorted(df['path'].tolist())

    os.makedirs(wav_target_dir, exist_ok=True)

    missing_list = []
    processed_list = []

    print("🔄 Starting preprocessing...\n")

    for filename in tqdm(sorted_list, desc="Processing audio"):
        original_wav = filename.replace(".mp3", file_extension)
        source_path = os.path.join(wav_source_dir, original_wav)

        if not os.path.exists(source_path):
            missing_list.append(filename)
            continue

        try:
            audio = AudioSegment.from_wav(source_path)

            # Normalize volume
            audio = match_target_amplitude(audio, -20.0)

            # Convert to mono and set sample rate
            audio = audio.set_channels(1).set_frame_rate(sample_rate)

            # Trim silence from beginning and end
            audio = trim_silence(audio)

            # Filter out too short or too long files
            if len(audio) < min_duration_ms or len(audio) > max_duration_ms:
                print(f"⏩ Skipped (duration): {filename}")
                continue

            # Save preprocessed file
            target_path = os.path.join(wav_target_dir, original_wav)
            audio.export(target_path, format="wav")
            processed_list.append(original_wav)

        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
            missing_list.append(filename)

    print(f"\n✅ Preprocessing done. {len(processed_list)} files saved.")
    print(f"⚠️ Missing or skipped: {len(missing_list)}")

    return missing_list, processed_list


def trim_silence(audio, silence_thresh=-40, padding=100):
    """
    Trims silence from beginning and end of the audio.
    Returns: trimmed AudioSegment
    """
    non_silents = detect_nonsilent(audio, min_silence_len=200, silence_thresh=silence_thresh)
    if not non_silents:
        return audio  # No nonsilent parts detected, return as-is

    start_trim = max(0, non_silents[0][0] - padding)
    end_trim = min(len(audio), non_silents[-1][1] + padding)

    return audio[start_trim:end_trim]


def match_target_amplitude(sound, target_dBFS):
    """
    Normalize volume to target dBFS.
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

