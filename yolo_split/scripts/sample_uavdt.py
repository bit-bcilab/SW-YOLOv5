

import json
import os
import random
import numpy as np


def sample_uavdt(json_path, sample_rate):
    with open(json_path, 'r') as f:
        whole_data = json.load(f)
        f.close()

    imgs = whole_data['imgs']
    num_all = len(imgs)
    all_indices = np.arange(0, num_all).tolist()

    videos = {}
    video_names = []
    video_frames = {}

    for i in range(num_all):
        img = imgs[i]
        video_name = img[:5]
        if video_name not in video_names:
            video_names.append(video_name)
            videos[video_name] = []
            video_frames[video_name] = []
        img_name = img[6:12]
        videos[video_name].append(i)
        video_frames[video_name].append(int(img_name))

    sampled_frames_ = []
    num_all_ = 0
    for i in range(len(video_names)):
        video_name = video_names[i]
        frame_indices = videos[video_name]
        frames = video_frames[video_name]
        num_frame = len(frame_indices)
        num_all_ += num_frame

        num_sample = int(num_frame / sample_rate)
        sample_indices = np.linspace(0, num_frame-1, num=num_sample, dtype=np.int64)
        frames_sorted = np.array(frames).argsort()
        frames_sorted_sample = frames_sorted[sample_indices]
        frame_indices = np.array(frame_indices)
        sampled_frames = frame_indices[frames_sorted_sample].tolist()
        sampled_frames_ += sampled_frames

    assert num_all_ == num_all

    sampled_imgs = [whole_data['imgs'][i] for i in sampled_frames_]
    sampled_labels = [whole_data['labels'][i] for i in sampled_frames_]
    sampled_labels_abs = [whole_data['labels_abs'][i] for i in sampled_frames_]
    sampled_shapes = [whole_data['shapes'][i] for i in sampled_frames_]

    sampled_data = {}
    sampled_data['dir'] = whole_data['dir']
    sampled_data['imgs'] = sampled_imgs
    sampled_data['labels'] = sampled_labels
    sampled_data['labels_abs'] = sampled_labels_abs
    sampled_data['shapes'] = sampled_shapes

    file = json.dumps(sampled_data, indent=4)
    fileObject = open('UAVDT-test-sampled.json', 'w')
    fileObject.write(file)
    fileObject.close()
    a = 0


if __name__ == "__main__":
    json_path = 'F://System/Desktop/UAVDT-test.json'
    sample_rate = 8
    sample_uavdt(json_path, sample_rate)
    pass
