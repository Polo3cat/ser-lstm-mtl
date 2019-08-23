import csv

from librosa.util import frame
from librosa.core import load



meta_files = ['iemocap2.txt', 'aibo2.txt', 'emodb2.txt', 'enterface2.txt', 'ldc2.txt']
#meta_files = ['tiny_dataset.txt']
frame_ms = 25
sliding_ms = 10
n_frames = 0
n_files = 0
for meta_file in meta_files:
    with open(meta_file) as f:
        for line in csv.DictReader(f, dialect='excel-tab'):
            filename = line.get('n_train_data.name')
            time_series, sr = load(filename)
            sr_ms = sr / 1000
            frames = frame(time_series, frame_length=int(sr_ms*frame_ms), hop_length=int(sr_ms*sliding_ms))
            n_frames += frames.shape[1]
            n_files += 1

print(f'Files: {n_files}')
print(f'Frames: {n_frames}')
