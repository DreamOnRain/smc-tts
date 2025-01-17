from funasr import AutoModel
import torch
import torchaudio

class funASR:
    def __init__(self, filename):
        self.model = AutoModel(model="paraformer-zh")
        self.file = filename
        self.fragments = None

    def get_timestamps_from_file(self, filename):
        raw_result = self.model.generate(filename)[0]
        characters = raw_result.get("text", "").split()
        timestamps = raw_result.get("timestamp", [])
        return characters, timestamps

    def cut_audio(self, start_time, end_time):
        waveform, sample_rate = torchaudio.load(self.file)
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        cut_waveform = waveform[:, start_sample:end_sample]
        temp_filename = "tmp.wav"
        torchaudio.save(temp_filename, cut_waveform, sample_rate)

        return temp_filename

    def get_timestamps_from_audio(self):
        self.fragments = self.file
        start_time, end_time = 0, 30
        temp_filename = self.cut_audio(start_time, end_time)
        characters, timestamps = self.get_timestamps_from_file(temp_filename)

        return characters, timestamps

funasr = funASR("agriculture_0000.wav")
characters, timestamps = funasr.get_timestamps_from_audio()
print(characters)