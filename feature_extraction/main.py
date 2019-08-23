'''
	Extract features and add them to a postgres database
'''
import psycopg2
import csv
from sys import exit

from librosa.core import load
from librosa.util import frame, pad_center
from librosa.feature import mfcc, delta, zero_crossing_rate
from scipy.integrate import trapz

from fundamental.fundamental import fundamental


emotion = {0: 'anger', 1: 'happiness', 2: 'sadness', 3: 'neutral'}
corpus = {0: 'enterface', 1: 'emodb', 2: 'aibo', 3: 'iemocap', 4: 'ldc'}

conn = psycopg2.connect('postgresql://docker:docker@localhost:5432/features')
cursor = conn.cursor()

meta_files = ['iemocap.txt', 'aibo.txt', 'emodb.txt', 'enterface.txt', 'ldc.txt']
#meta_files = ['tiny_dataset.txt']

min_max = dict(zip(meta_files, [(0,0)]*len(meta_files)))
counter = 0

'''
	First pass, we insert the labels into the database and calculate the minimum and maximum values of signals
	in each separate dataset in order to normalize them.
	It is important to do this for each dataset separately since they contain distinct recording conditions
	and gains.
'''
for meta_file in meta_files:
	with open(meta_file) as f:
		for line in csv.DictReader(f, dialect='excel-tab'):
			filename = line.get('n_train_data.name')
			'''
				Insert the labels data into "labels" table. Later it will be used as a FK for the window level feature.
			'''
			cursor.execute('''INSERT INTO labels (filepath, gender, acted, emotion, arousal, valence, speaker_number, corpus)
				VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
				ON CONFLICT DO NOTHING;
				''',[	filename,
						line.get('gender'),
						line.get('acted'),
						emotion[int(line.get('class'))],
						int(line.get('arousal')),
						int(line.get('valence')),
						int(line.get('sid')),
						corpus[int(line.get('corpus_id'))]])
			# Process filename
			time_series, sr = load(filename)
			sr_ms = sr / 1000

			min_ = time_series.min()
			max_ = time_series.max()
			min_max[meta_file] = (min(min_max[meta_file][0], min_), max(min_max[meta_file][1], max_))

			counter += 1
			if not counter%20:
				conn.commit()
				print(counter)

print(min_max)
'''
	Second pass, we extract the window level features and insert them into the database.
'''
frame_ms = 25
sliding_ms = 10
counter = 0
for meta_file in meta_files:
	with open(meta_file) as f:
		for line in csv.DictReader(f, dialect='excel-tab'):
			filename = line.get('n_train_data.name')
			time_series, sr = load(filename)
			sr_ms = sr / 1000
			'''
				Zero crossing rates and fundamental frequencies must be computed before normalizing
				the data, otherwise we are not calculating what we actually want.
				For ZCR no value crosses 0 after normalizing and the fundamentals won't
				correspond to the actual frequencies in hertz.
			'''
			zero_crossing_rates = zero_crossing_rate(time_series, 
														frame_length=int(frame_ms*sr_ms), 
														hop_length=int(sliding_ms*sr_ms),
														center=True)
			frames = frame(time_series, frame_length=int(sr_ms*frame_ms), hop_length=int(sr_ms*sliding_ms))
			frames = pad_center(frames, size=zero_crossing_rates.shape[1], axis=1)
			fundamentals = fundamental(frames, sr)
			'''
				We normalize with respect to the maximum and minimum found across the corpus.
			'''
			time_series = (time_series - min_max[meta_file][0]) / (min_max[meta_file][1] - min_max[meta_file][0])
			mfccs = mfcc(time_series,
							sr=sr,
							n_mfcc=12,
							n_fft=int(frame_ms*sr_ms),
							hop_length=int(sliding_ms*sr_ms))
			d_mfccs = delta(mfccs, width=3, order=1)

			frames = frame(time_series, frame_length=int(sr_ms*frame_ms), hop_length=int(sr_ms*sliding_ms))
			frames = pad_center(frames, size=mfccs.shape[1], axis=1)
			energies = trapz(frames*frames, dx=frame_ms, axis=0)

			for instant, (f0, zcr, e, frame_mfccs, frame_delta_mfccs) in enumerate(zip(fundamentals, 
																					zero_crossing_rates.T,
																					energies, 
																					mfccs.T,
																					d_mfccs.T)):
				cursor.execute('''WITH fn (label_id) AS (
					SELECT id FROM labels WHERE filepath = %s LIMIT 1)
					INSERT INTO frames (instant, f0, zcr, energy, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, delta_mfcc1, delta_mfcc2, delta_mfcc3, delta_mfcc4, delta_mfcc5, delta_mfcc6, delta_mfcc7, delta_mfcc8, delta_mfcc9, delta_mfcc10, delta_mfcc11, delta_mfcc12, label_)
					VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s, (SELECT label_id FROM fn))
					ON CONFLICT DO NOTHING;
					''', (filename, instant, float(f0), float(zcr[0]), float(e), *(float(x) for x in frame_mfccs), *(float(x) for x in frame_delta_mfccs)))
			
			counter += 1
			if not counter%20:
				conn.commit()
				print(counter)
	conn.commit()
