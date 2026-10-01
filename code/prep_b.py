#!/usr/bin/env python3
"""Part B, step 1: separate each ad into clean channels.

V  : muted video                        out/var/<id>_V.mp4
P  : prosody audio (low-pass 400 Hz)    out/var/<id>_P.wav   pitch/rhythm/energy kept, words unintelligible
VP : video + prosody audio              out/var/<id>_VP.mp4
A  : raw audio (speech + music)         out/audio/<id>.wav   (from poc_clip.extract_audio)
T  : transcript (faster-whisper words)  out/transcripts/<id>.json
"""
import csv, json, os, subprocess
from poc_clip import DATA, OUT, extract_audio, extract_frames, transcribe

VAR, TR = os.path.join(OUT, "var"), os.path.join(OUT, "transcripts")
LOWPASS = "lowpass=f=400,lowpass=f=400,loudnorm"  # two poles -> steeper roll-off; loudnorm equalises level


def sh(cmd):
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def main():
    os.makedirs(VAR, exist_ok=True)
    os.makedirs(TR, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.csv"))))
    for r in rows:
        vid, src = r["id"], os.path.join(DATA, r["video"])
        extract_frames(vid, src)
        wav = extract_audio(vid, src)
        v, p, vp = (os.path.join(VAR, f"{vid}_{k}") for k in ("V.mp4", "P.wav", "VP.mp4"))
        if not os.path.exists(v):
            sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-an", "-c:v", "copy", v])
        if not os.path.exists(p):
            sh(["ffmpeg", "-v", "error", "-y", "-i", wav, "-af", LOWPASS, "-ac", "1", "-ar", "16000", p])
        if not os.path.exists(vp):
            sh(["ffmpeg", "-v", "error", "-y", "-i", v, "-i", p, "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-shortest", vp])
        tj = os.path.join(TR, vid + ".json")
        if not os.path.exists(tj):
            words = transcribe(wav)
            json.dump({"words": words, "text": " ".join(w for _, w in words)}, open(tj, "w"))
        print(f"{vid}: ok ({len(json.load(open(tj))['words'])} words)")


if __name__ == "__main__":
    main()
