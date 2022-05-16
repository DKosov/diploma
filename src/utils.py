import os
import cv2
import random
import librosa
import numpy as np
from tqdm import tqdm
from collections import defaultdict


AUDIO_EXTENSIONS = ['mp3', 'wav', 'aac']
VIDEO_EXTENSIONS = ['mov', 'mp4']

tempo_sensitivity = 0.25
shake = 0.5
truncation = 1


def process_music(audio_file, fps, all_feats=False):
    y, sr = librosa.load(audio_file)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=127, fmax=8000, hop_length=round(sr / fps))
    specm = np.mean(spec, axis=0)
    gradm = np.gradient(specm)
    gradm = gradm / np.max(gradm)
    gradm = gradm.clip(min=0)

    audio_feats = np.concatenate((spec, gradm[np.newaxis, :]), axis=0).transpose()

    if all_feats:
        return spec, specm, gradm, audio_feats
    else:
        return audio_feats


def new_shakes(shake):
    shakes = np.zeros(128)
    for j in range(128):
        if random.uniform(0, 1) < 0.5:
            shakes[j] = 1
        else:
            shakes[j] = 1 - shake        
    return shakes


def new_update_vec(nv2, update_vec):
    for ni, n in enumerate(nv2):                  
        if n >= (2 * truncation - tempo_sensitivity):
            update_vec[ni] = -1
        elif n < (-2 * truncation + tempo_sensitivity):
            update_vec[ni] = 1   
    return update_vec


def musicality_noise_vectors(specm, gradm, dim=128):
    nv1 = np.random.rand(dim)
    noise_vectors = [nv1]
    nvlast = nv1
    update_vec = np.zeros(128)
    for ni, n in enumerate(nv1):
        if n < 0:
            update_vec[ni] = 1
        else:
            update_vec[ni] = -1

    update_last = np.zeros(128)
    
    for i in tqdm(range(len(gradm))):      
        if (i % 200) == 0:
            shakes = new_shakes(shake)
        nv1 = nvlast
        update = np.array([tempo_sensitivity for _ in range(128)]) * (gradm[i]+specm[i]) * update_vec * shakes 
        update = (update + update_last * 3) / 4
        update_last = update
        nv2 = nv1 + update
        noise_vectors.append(nv2)
        nvlast = nv2
        update_vec = new_update_vec(nv2, update_vec)

    return noise_vectors


def process_video(video_file, fps=None, frame_size = (256, 256)):
    video = cv2.VideoCapture(video_file)
    
    v_fps = video.get(cv2.CAP_PROP_FPS)
    fps = v_fps

    all_frames = []
    i_cnt = 0
    if fps is not None:
        skip_length = max(round(v_fps/fps), 1)

    while True:
        i_cnt += 1
        ret, frame = video.read()

        if ret == False:
            break
        if (fps is not None) and ((i_cnt % skip_length) != 0):
            continue

        frame = cv2.resize(frame, (frame_size))
        all_frames.append(frame)

    return np.asarray(all_frames), fps


def create_train_dataset(video_files, audio_files, base_folder=""):
    audio_video_dict = dict()
    for v_file, a_file in zip(video_files, audio_files):
        video_frames, fps = process_video(os.path.join(base_folder, v_file), fps = None)
        spec, specm, gradm, audio_features = process_music(os.path.join(base_folder, a_file), fps = fps, all_feats=True)
        noise_vectors = musicality_noise_vectors(specm, gradm)
        frame_count = min(len(video_frames), len(audio_features))
        video_frames = video_frames[:frame_count]
        audio_features = audio_features[:frame_count]
        noise_vectors = audio_features[:frame_count]
        audio_video_dict[v_file]=(audio_features, video_frames, noise_vectors)
    return audio_video_dict


def get_audio_video_frame_data(folder):
    audio_video_files = os.listdir(folder)
    audio_video_pairing = defaultdict(dict)
    for f in audio_video_files:
        f_name = f.split(".")[0]
        f_ext = f.split(".")[-1]

        if f_ext in AUDIO_EXTENSIONS:
            audio_video_pairing[f_name]["audio"] = f

        if f_ext in VIDEO_EXTENSIONS:
            audio_video_pairing[f_name]["video"] = f

    all_video_files = []
    all_audio_files = []
    for f in audio_video_pairing:
        all_video_files.append(audio_video_pairing[f]["video"])
        all_audio_files.append(audio_video_pairing[f]["audio"])
        
    audio_video_features = create_train_dataset(all_video_files, all_audio_files, base_folder=folder)

    return audio_video_features
