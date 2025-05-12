import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

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





