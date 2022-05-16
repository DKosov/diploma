import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from src.utils import get_audio_video_frame_data


class MusicWithFrames(data.Dataset):
    def __init__(self, folder=None, transform=None):
        music_frame_data = get_audio_video_frame_data(folder)
        self.audio2frames = []
        for key in music_frame_data:
            audio_feats, video_frames, noise_vectors = music_frame_data[key]
            entry=[]
            for a_f, v_f, n_f in zip(audio_feats, video_frames, noise_vectors):
                entry.append((a_f, v_f, n_f))
                entry2 = list(zip(*entry))
                gaussian = np.linspace(-3,3, len(entry2[0]))
                gaussian = np.exp(-gaussian*gaussian)
                gaussian = np.repeat(gaussian[:, np.newaxis], 128, axis=1)
                weighted_audio_feat= np.sum(gaussian*entry2[0], axis=0)

                self.audio2frames.append((weighted_audio_feat, [entry2[1][int(len(entry)/2)]], entry2[2][0]))
                if len(entry) == 5:
                    entry=entry[1:] 
        
        self.transform = transform

    def __getitem__(self, idx):
        audio_feats, video_frames, noise_vectors = self.audio2frames[idx]
        video_frames = [Image.fromarray(video_frame) for video_frame in video_frames]
        if self.transform != None:
            video_frames = [self.transform(video_frame) for video_frame in video_frames]
        new_video_frame = torch.stack(video_frames, dim=0)
        size = new_video_frame.size()
        new_video_frame = new_video_frame.view(-1,size[-2], size[-1])
        return new_video_frame, audio_feats, noise_vectors

    def __len__(self):
        return len(self.audio2frames)
