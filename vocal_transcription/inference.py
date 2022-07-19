import numpy as np
import librosa
import torch
import time
import sys
sys.path.insert(0, './vocal_transcription')

import config
from model import VocalNet
from utils import move_data_to_device, append_to_dict, RegressionPostProcessor, write_events_to_midi

class VocalTranscritiopn(object):
    def __init__(self, checkpoint_path=None) -> None:
        if checkpoint_path is None:
            self.checkpoint_path = "checkpoints/2022-05-09-19-59-51/19999_iterations.pth"
        else:
            self.checkpoint_path = checkpoint_path
        self.segment_samples = int(config.sample_rate * config.segment_seconds)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VocalNet(frames_per_second=config.frames_per_second,
                              classes_num=config.classes_num)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)



    def transcribe(self, audio, midi_path, instrument_id, batch_size=1):
        """
        [input]
            audio:          numpy array (audio wav) | str (audio path)
            midi_path:      output path of midi file
            batch_size:     batch_size of input data
        """
        bgn_time = time.time()
        if isinstance(audio, str):  # if it's path of the audio
            audio, _ = librosa.core.load(audio, sr=16000)
        segments = self.enframe(audio)
        output_dict = self.forward(segments, batch_size=batch_size)
        output_dict = self.deframe(output_dict, len(audio))
        post_processor = RegressionPostProcessor(frames_per_second=config.frames_per_second, 
                                                 classes_num=config.classes_num, 
                                                 onset_threshold=0.3, 
                                                 offset_threshold=0.3, 
                                                 frame_threshold=0.1, 
                                                 pedal_offset_threshold=0.2)
        (est_note_events, est_pedal_events) = \
            post_processor.output_dict_to_midi_events(output_dict)
        if midi_path:
            write_events_to_midi(start_time=0, note_events=est_note_events, 
                pedal_events=est_pedal_events, midi_path=midi_path, instrument_id=instrument_id)
            print('Write out to {}'.format(midi_path))
            
        transcribed_dict = {
            'output_dict': output_dict, 
            'est_note_events': est_note_events,
            'est_pedal_events': est_pedal_events}

        cost_time = time.time() - bgn_time
        print("cost time: %dmin %ds"%(cost_time//60, cost_time%60))

        return transcribed_dict


    def enframe(self, audio):
        # padding
        audio = audio[None, :]
        len_audio = audio.shape[1]
        if len_audio % self.segment_samples != 0:
            len_pad = int(self.segment_samples - (len_audio % self.segment_samples))
            audio = np.concatenate((audio, np.zeros((1, len_pad))), axis=1)
        
        # segmentting
        batch = []
        pointer = 0
        while pointer + self.segment_samples <= audio.shape[1]:
            batch.append(audio[:, pointer : pointer + self.segment_samples])
            pointer += self.segment_samples // 2
        batch = np.concatenate(batch, axis=0)

        return batch


    def deframe(self, output_dict, len_audio):
        for key, value in output_dict.items():
            if value.shape[0] == 1:
                output_dict[key] = value[0][:len_audio]
            else:
                value = value[:, 0: -1, :]
                (N, segment_samples, _) = value.shape
                assert segment_samples % 4 == 0

                y = []
                y.append(value[0, 0 : int(segment_samples * 0.75)])
                for i in range(1, N - 1):
                    y.append(value[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
                y.append(value[-1, int(segment_samples * 0.25) :])
                y = np.concatenate(y, axis=0)
                output_dict[key] = y[:len_audio]
        return output_dict


    def forward(self, x, batch_size):
        """Forward data to model in mini-batch. 
        Args: 
        model: object
        x: (N, segment_samples)
        batch_size: int

        Returns:
        output_dict: dict, e.g. {
            'frame_output': (segments_num, frames_num, classes_num),
            'onset_output': (segments_num, frames_num, classes_num),
            ...}
        """
        output_dict = {}
        device = next(self.model.parameters()).device
        pointer = 0
        while True:
            if pointer >= len(x):
                break

            batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
            pointer += batch_size

            with torch.no_grad():
                self.model.eval()
                batch_output_dict = self.model(batch_waveform)
            for key in batch_output_dict.keys():
                # if '_list' not in key:
                append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict


if __name__ == '__main__':
    transriber = VocalTranscritiopn()
    bgn_time = time.time()
    transriber.transcribe('test/408/Vocal.wav', 'mm.mid')
    cost = time.time() - bgn_time
    print("cost: %dmin%ds"%(cost//60, cost%60))