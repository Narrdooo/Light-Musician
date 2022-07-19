import librosa
import os
import json
import numpy as np

import librosa
import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, './')


class ST500_Dataset(object):
    def __init__(self, segment_seconds, frames_per_second, sample_rate, classes_num) -> None:
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * self.sample_rate)
        self.convert_ratio = sample_rate / frames_per_second
        self.n_notes = classes_num
        with open('./MIR-ST500_20210206/MIR-ST500_corrected.json', 'r') as f:
            self.label_dict = json.load(f)


    def __getitem__(self, meta):
        data_id, audio_path, bgn_time = meta
        bgn_sample = int(bgn_time * self.sample_rate)
        end_sample = bgn_sample + self.segment_samples
        
        (audio, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        audio = audio[bgn_sample: end_sample]

        target_dict = self.label_process(data_id, bgn_sample, end_sample)
        
        data_dict = target_dict
        data_dict['waveform'] = audio
        return data_dict


    def label_process(self, data_id, bgn_sample, end_sample):
        """
        return: 
            target_dict:{
                'onset_roll': (frames_num, n_notes), 
                'offset_roll': (frames_num, n_notes), 
                'reg_onset_roll': (frames_num, n_notes), 
                'reg_offset_roll': (frames_num, n_notes), 
                'frame_roll': (frames_num, n_notes), 
                'mask_roll':  (frames_num, n_notes), 
            }
        """
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.n_notes))
        offset_roll = np.zeros((frames_num, self.n_notes))
        reg_onset_roll = np.ones((frames_num, self.n_notes))
        reg_offset_roll = np.ones((frames_num, self.n_notes))
        frame_roll = np.zeros((frames_num, self.n_notes))
        mask_roll = np.ones((frames_num, self.n_notes))

        labels = self.label_dict[data_id]

        for bgn_time, end_time, note in labels:
            note = int(note)
            b_sample = bgn_time * self.sample_rate
            e_sample = end_time * self.sample_rate

            if b_sample >= end_sample or e_sample <= bgn_sample:
                continue 

            bgn_frame = int((b_sample - bgn_sample) / self.convert_ratio)
            end_frame = int((e_sample - bgn_sample) / self.convert_ratio)
            
            frame_roll[max(bgn_frame, 0): min(end_frame+1, frames_num), note] = 1

            if bgn_frame >= 0:              
                onset_roll[bgn_frame, note] = 1
                reg_onset_roll[bgn_frame, note] = \
                    ((b_sample - bgn_sample) / self.sample_rate) - (bgn_frame / self.frames_per_second)
            else:
                mask_roll[: end_frame+1, note] = 0

            if end_frame < frames_num:      
                offset_roll[end_frame, note] = 1
                reg_offset_roll[end_frame, note] = \
                    ((e_sample - bgn_sample) / self.sample_rate) - (end_frame / self.frames_per_second)
            else:
                mask_roll[bgn_frame: , note] = 0

        for k in range(self.n_notes):
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        target_dict = {
            'onset_roll': onset_roll, 
            'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 
            'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 
            'mask_roll': mask_roll, 
        }
        return target_dict

            
    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output



class Sampler(object):
    def __init__(self, data_dir, segment_seconds, hop_seconds, sample_rate, 
            batch_size, type, random_seed=1234):

        self.data_dir = data_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        self.segment_list = []
        for data_id in os.listdir(self.data_dir):
            audio_path = os.path.join(self.data_dir, data_id, "Vocal.wav")
            wav, sr = librosa.core.load(audio_path, sr=None)
            len_time = len(wav)//sr
            bgn_time = 0
            while(bgn_time + self.segment_seconds < len_time):
                self.segment_list.append([data_id, audio_path, bgn_time])
                bgn_time += self.hop_seconds


        print("%s samples:%d"%(type, len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']



class TestSampler(Sampler):
    def __init__(self, data_dir, segment_seconds, hop_seconds, sample_rate, 
            batch_size, type, random_seed=1234):
        super().__init__(data_dir, segment_seconds, hop_seconds, sample_rate, 
            batch_size, type, random_seed)
        

        # logging.info('Evaluate segments: {}'.format(len(self.segment_list)))


    def __iter__(self):
        p = 0
        iteration = 0
        self.max_evaluate_iteration = 5    # Number of mini-batches to validate
        while True:
            if iteration == self.max_evaluate_iteration:
                break

            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[p]
                p += 1

                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1
            yield batch_segment_list



def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, n_notes), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, n_notes), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict

