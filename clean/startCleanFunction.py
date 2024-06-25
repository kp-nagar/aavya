import cv2
from tqdm import tqdm
from os import path
import os
import ffmpeg
import cv2
import time


def convert_frame(videoname, outputdir):
  inputVideoPath = f'results/{videoname}'
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
  voicePath = f"voice/{videoName}.wav"
  video_name = f"{videoName}.mp4"
  input_video = ffmpeg.input(videoPath)
  input_audio = ffmpeg.input(voicePath)
  outputCleanPath = f"uploads/{video_name}"
  ffmpeg.concat(input_video, input_audio, v=1, a=1).output(outputCleanPath).run()
  return video_name


async def clean_video(videoname: str):
    start_time = time.time()
    outputdir = videoname.split(".")[0]
    convert_frame(videoname, outputdir)
    print(f"convert_frame: {time.time()-start_time}")
    start_time = time.time()
    cleanframeDir = f"res_{outputdir}"
    os.system(f'python clean/inference_gfpgan.py -i {outputdir} -o {cleanframeDir} -v 1.4 -s 2 --only_center_face --bg_upsampler None')
    print(f"inference_gfpgan: {time.time()-start_time}")
    start_time = time.time()
    combine_frames(cleanframeDir, outputdir)
    print(f"combine_frames: {time.time()-start_time}")
    start_time = time.time()
    videoPath = f'{cleanframeDir}/{outputdir}.avi'
    video_path = combine_voice_video(outputdir, videoPath)
    print(f"combine_voice_video: {time.time()-start_time}")
    return video_path
