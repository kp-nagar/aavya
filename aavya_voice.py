from fastapi import FastAPI, BackgroundTasks
from whisperspeech.pipeline import Pipeline
import time
import os
from clean.startCleanFunction import clean_video
import torch

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')


def clean_data(id):
    print(f"removing id: {id}")
    os.system(f'rm -r {id} res_{id}')


@app.post("/")
async def text_speech(tts: str, background_tasks: BackgroundTasks):
    start_time = time.time()
    fileid = int(time.time_ns())
    print(f"-----------------------{fileid}")
    filename = f"{fileid}.wav"
    filepath = f"voice/{filename}"
    pipe.generate_to_file(filepath, tts, cps=13, lang='en')
    print(f"generate_to_file: {time.time()-start_time}")
    start_time = time.time()
    video_lip_filepath = f"results/{fileid}.mp4"
    face_video = "talk.mp4"
    os.system(f'python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {face_video} --audio {filepath} --outfile {video_lip_filepath}')
    print(f"inference: {time.time()-start_time}")
    start_time = time.time()
    await clean_video(videoname=f"{fileid}.mp4")
    print(f"clean_video: {time.time()-start_time}")
    background_tasks.add_task(clean_data, fileid)
    return {"filename": filename, "filepath": filepath}
