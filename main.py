# -*- coding: utf-8 -*-

import supervision as sv
import numpy as np
import rgb_array #use this only if we get the rest of it working lol.
from ultralytics import YOLO

#-------- model and video information. --------#
VIDEO_PATH = "video.mp4" #replace this with video path.
model = YOLO("best.pt") #replace this with model (.pt file).

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

#-------- process every frame in the video --------#
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0] #imports the frame for processing.
    
    segmentations = results.pred.segmentation #retrieves segmented mask information from yolov8 model.
    
    masks = [s[c].numpy() for s in segmentations]
    
    #creates binary pixel array for mask.
    binary_masks = []
    for i, mask in enumerate(masks):
        binary_mask = (masks == i)
        binary_masks.append(binary_mask)
        
    segmented_frame = frame.copy()
    for mask in binary_masks:
        segmented_frame[~mask] = 0;
    
    return frame

sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
