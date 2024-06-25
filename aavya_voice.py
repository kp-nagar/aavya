from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from whisperspeech.pipeline import Pipeline
import time
import os
from fastapi.middleware.cors import CORSMiddleware
from clean.startCleanFunction import clean_video
from fastapi.staticfiles import StaticFiles
from moviepy.editor import ImageClip
# import torch
import shutil


app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the correct origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-small-en+pl.model')


def clean_data(id):
    print(f"removing id: {id}")
    os.system(f'rm -r {id} res_{id}')


def bark_tts(ttx, path):
    from transformers import BarkModel
    model = BarkModel.from_pretrained("suno/bark-small")
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("suno/bark")
    # prepare the inputs
    inputs = processor(ttx, voice_preset="v2/en_speaker_9")
    # generate speech
    speech_output = model.generate(**inputs.to(device))
    import scipy
    sampling_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(path, rate=sampling_rate, data=speech_output[0].cpu().numpy())


@app.post("/")
async def text_speech(tts: str, face_media_path: str, background_tasks: BackgroundTasks):
    start_time = time.time()
    fileid = int(time.time_ns())
    print(f"-----------------------{fileid}")
    filename = f"{fileid}.wav"
    filepath = f"voice/{filename}"
    pipe.generate_to_file(filepath, tts, cps=13, lang='en')
    print(f"generate_to_file: {time.time()-start_time}")
    start_time = time.time()
    video_lip_filepath = f"results/{fileid}.mp4"
    # face_media_path = "upload"
    os.system(f'python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {face_media_path} --audio {filepath} --outfile {video_lip_filepath}')
    print(f"inference: {time.time()-start_time}")
    start_time = time.time()
    await clean_video(videoname=f"{fileid}.mp4")
    print(f"clean_video: {time.time()-start_time}")
    background_tasks.add_task(clean_data, fileid)
    return {"filename": filename, "filepath": filepath}


async def text_speech(fileid, tts: str, face_media_path: str):
    start_time = time.time()
    print(f"-----------------------{fileid}, face_media_path: {face_media_path}")
    filename = f"{fileid}.wav"
    filepath = f"voice/{filename}"
    pipe.generate_to_file(filepath, tts, cps=13, lang='en')
    # bark_tts(tts, filepath)
    print(f"generate_to_file: {time.time()-start_time}")
    start_time = time.time()
    video_lip_filepath = f"results/{fileid}.mp4"
    # face_media_path = "upload"
    os.system(f'python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {face_media_path} --audio {filepath} --outfile {video_lip_filepath}')
    print(f"inference: {time.time()-start_time}")
    start_time = time.time()
    # video_path = await clean_video(videoname=f"{fileid}.mp4")
    print(f"clean_video: {time.time()-start_time}")
    video_path = "1.mp4"
    return video_path

upload_dir = "uploads"
app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

def create_video_from_image(image_path, video_path, duration=5):
    """Create a video from a single image."""
    clip = ImageClip(image_path)
    clip = clip.set_duration(duration)
    clip = clip.resize(newsize=(1280, 720))
    clip.write_videofile(video_path, fps=24)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), text: str = Form(...)):
    fileid = int(time.time_ns())
    upload_file_path = f"uploads/{fileid}"
    os.makedirs(upload_file_path, exist_ok=True)
    # Save the uploaded image
    image_path = os.path.join(upload_file_path, file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create video from the uploaded image
    video_filename = file.filename.rsplit('.', 1)[0] + '.mp4'
    video_path = os.path.join(upload_file_path, video_filename)
    create_video_from_image(image_path, video_path)
    video_path = await text_speech(fileid=fileid, tts=text, face_media_path=video_path)
    return {"filename": video_path}
