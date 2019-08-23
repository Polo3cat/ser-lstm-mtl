'''
	In order to train on the UT's HPC I need to store the features
	directly in a file, as opposed to loading from a postgres database
	like I originally was doing locally.
	Therefore, we take the FrameSequence output, that would usually
	be fed into the keras trainer, and store it to later feed it
	to the trainer directly.
'''
import pickle
import argparse

import psycopg2
import numpy as np
from keras.utils import to_categorical


parser = argparse.ArgumentParser()
parser.add_argument('save_dir', help='Directory where to save the pickled features and labels')
args = parser.parse_args()

conn = psycopg2.connect('postgresql://docker:docker@localhost:5432/features')

emotion_map = {'anger': 0, 'happiness': 1, 'sadness': 2, 'neutral': 3}

class FramesLabelsIter:
    def __init__(self, ids):
        self.ids = list(ids)
        self.cursor = conn.cursor()
        self.sentinel = 0

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        print(self.sentinel)
        if self.sentinel >= len(self.ids):
            raise StopIteration

        id_ = self.ids[self.sentinel]
        self.sentinel += 1
        self.cursor.execute('''SELECT gender, acted, emotion FROM labels
                            WHERE id = %s''', (id_,)) 
        gender, acted, emotion = self.cursor.fetchone()
        categorical_labels = [  to_categorical(gender,num_classes=2).reshape(1,2), 
                                to_categorical(acted,num_classes=2).reshape(1,2), 
                                to_categorical(emotion_map[emotion],num_classes=4).reshape(1,4)]
        self.cursor.execute('''SELECT f0, zcr, energy, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, 
                         mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, delta_mfcc1, delta_mfcc2, delta_mfcc3, 
                         delta_mfcc4, delta_mfcc5, delta_mfcc6, delta_mfcc7, delta_mfcc8, delta_mfcc9, delta_mfcc10, 
                         delta_mfcc11, delta_mfcc12
                    FROM frames WHERE label_ = %s ORDER BY instant''', (id_,))
        n_frames = self.cursor.rowcount
        frames = np.empty((1, n_frames, 27))
        for i, frame in enumerate(self.cursor):
            frames[0,i] = np.array(frame)
        return id_, frames, categorical_labels

cursor = conn.cursor()
cursor.execute('''SELECT id FROM labels''')

'''
      Reading of all ids
'''
sequence = FramesLabelsIter((x[0] for x in cursor))
output_dict = dict()
for id_, frames, categorical_labels in sequence:
	output_dict[id_] = {'frames': frames, 'labels': categorical_labels}
pickle.dump(output_dict, f"{args.save_dir}/frames_labels.pkl", protocol=4)
