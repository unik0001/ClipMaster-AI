
import os
import subprocess
import numpy as np
import wave
import argparse

def run(cmd):
    subprocess.run(cmd, check=True)

def get_audio(video, wav_file):
    run([
        "ffmpeg", "-y", "-i", video,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", wav_file
    ])

def analyse_audio(wav_file):
    with wave.open(wav_file, "rb") as f:
        rate = f.getframerate()
        audio = np.frombuffer(
            f.readframes(f.getnframes()),
            dtype=np.int16
        ).astype(np.float32)

    audio /= 32768.0
    seconds = len(audio) / rate
    rms = []

    for i in range(int(seconds)):
        chunk = audio[i*rate:(i+1)*rate]
        rms.append(
            np.sqrt(np.mean(chunk**2)) if len(chunk) else 0
        )

    return np.array(rms), seconds

def select_moments(rms, duration, clip_length=30, count=5):
    candidates = []

    for start in range(0, max(1, int(duration-clip_length)+1), 5):
        end = min(start+clip_length, len(rms))
        candidates.append((np.mean(rms[start:end]), start))

    candidates.sort(reverse=True)
    selected = []

    for score, start in candidates:
        if all(abs(start-old) >= clip_length for old in selected):
            selected.append(start)
        if len(selected) >= count:
            break

    return sorted(selected)

def create_clip(video, start, length, output):
    run([
        "ffmpeg", "-y", "-ss", str(start),
        "-i", video, "-t", str(length),
        "-map", "0:v:0", "-map", "0:a?",
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920",
        "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-c:a", "aac",
        "-movflags", "+faststart", output
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--length", type=int, default=30)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    wav_file = "temp_audio.wav"

    print("Analysing video...")
    get_audio(args.video, wav_file)
    rms, duration = analyse_audio(wav_file)

    if duration < args.length:
        raise ValueError("Video is too short.")

    moments = select_moments(
        rms, duration, args.length, args.clips
    )

    for i, start in enumerate(moments, 1):
        print("Creating clip", i)
        create_clip(
            args.video, start, args.length,
            f"output/clip_{i:02d}.mp4"
        )

    os.remove(wav_file)
    print("Finished! Check output folder.")

if __name__ == "__main__":
    main()
