import cv2
from tqdm import tqdm
from os import path
from fastapi import FastAPI
import os
import ffmpeg
import cv2


app = FastAPI()


def convert_frame(videoname, outputdir):
  inputVideoPath = f'../results/{videoname}'
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)

  vidcap = cv2.VideoCapture(inputVideoPath)
  numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  print("FPS: ", fps, "Frames: ", numberOfFrames)

  for frameNumber in tqdm(range(numberOfFrames)):
      _,image = vidcap.read()
      cv2.imwrite(path.join(outputdir, str(frameNumber).zfill(4)+'.jpg'), image)


def combine_frames(cleanFramDir, outputFileName):
  restoredFramesPath = cleanFramDir + '/restored_imgs/'

  dir_list = os.listdir(restoredFramesPath)
  dir_list.sort()

  batchSize = 300
  from tqdm import tqdm
  for i in tqdm(range(0, len(dir_list), batchSize)):
    img_array = []
    start, end = i, i+batchSize
    print("processing ", start, end)
    for filename in  tqdm(dir_list[start:end]):
        filename = restoredFramesPath+filename;
        img = cv2.imread(filename)
        if img is None:
          continue
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(f'{cleanFramDir}/{outputFileName}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
  
    for i in range(len(img_array)):
      out.write(img_array[i])
    out.release()


def combine_voice_video(videoName, videoPath):
  voicePath = f"../voice/{videoName}.wav"
  input_video = ffmpeg.input(videoPath)
  input_audio = ffmpeg.input(voicePath)
  outputCleanPath = f"../{videoName}.mp4"
  ffmpeg.concat(input_video, input_audio, v=1, a=1).output(outputCleanPath).run()


@app.post("/clean")
async def clean_video(videoname: str):
    outputdir = videoname.split(".")[0]
    convert_frame(videoname, outputdir)
    cleanframeDir = f"res_{outputdir}"
    os.system(f'python inference_gfpgan.py -i {outputdir} -o {cleanframeDir} -v 1.3 -s 2 --only_center_face --bg_upsampler None')
    combine_frames(cleanframeDir, outputdir)
    videoPath = f'{cleanframeDir}/{outputdir}.avi'
    combine_voice_video(outputdir, videoPath)
    return {"Done": True}