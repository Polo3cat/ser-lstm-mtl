import logging
import argparse
from collections import defaultdict
import pickle

import numpy as np
from keras.utils import Sequence, to_categorical
from keras.callbacks import EarlyStopping, ProgbarLogger, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from high_level_model import generate_lstm_model
from elm import ELM


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Train file')
    parser.add_argument('validation', help='Validation file')
    parser.add_argument('test', help='Test file')
    parser.add_argument('features', help='Pickled dictionary contatining features')
    parser.add_argument('save_dir', help='Directory where to save the model weights')
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('gender_weight', type=float)
    parser.add_argument('acted_weight', type=float)
    parser.add_argument('private_layers', help='Number of private layers on the tasks', type=int)
    parser.add_argument('prefix', help='Prefix added to all saved files for this training')
    args = parser.parse_args()

    f = open(args.features, 'rb')

    class FrameSequence(Sequence):
        frames_labels = pickle.load(f)

        def __init__(self, ids):
            self.ids = list(ids)

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return self.frames_labels[self.ids[idx]]['frames'], self.frames_labels[self.ids[idx]]['labels']


    f.close()
    '''
      Start of training reading
    '''
    f = open(args.train)
    labels_ids = (int(x) for x in f if x != '\n')
    training_sequence = FrameSequence(labels_ids)
    f.close()
    '''
      Start of validation reading
    '''
    f = open(args.validation)
    labels_ids = tuple((int(x) for x in f if x != '\n'))
    validation_sequence = FrameSequence(labels_ids)
    f.close()
    '''
      Start of test reading
    '''
    f = open(args.test)
    labels_ids = tuple((int(x) for x in f if x != '\n'))
    test_sequence = FrameSequence(labels_ids)
    f.close()

    model = generate_lstm_model(args.learning_rate, args.gender_weight, args.acted_weight, args.private_layers)
    history = model.fit_generator(training_sequence, 
                                steps_per_epoch=None, 
                                epochs=50, 
                                verbose=1, 
                                validation_data=validation_sequence, 
                                use_multiprocessing=False,
                                callbacks=[EarlyStopping(monitor='val_emotion_acc', patience=3, mode='max', restore_best_weights=True),
                                           ModelCheckpoint(args.save_dir + "/weights.{epoch:02d}-{val_emotion_acc:.4f}.hdf5",
                                                           save_best_only=True,
                                                           save_weights_only=True)])

    f = open(args.save_dir+f'/{args.prefix}_history.pkl', 'wb')
    pickle.dump(history, f, protocol=4)
    f.close()
    test_result = zip(model.metrics_names, model.evaluate_generator(test_sequence))
    with open(args.save_dir + f'/{args.prefix}_test_resutls.txt', mode='w') as f:
        f.write(str(list(test_result)))
    print('LSTM training and testing done.')
    print('Starting high level features plus ELM.')

    def extract_high_level_features(sequence):
        '''
            Extract high level features from sequence.
            For each of the emotions (columns in the arrays) we seek for the
            maximum, the minimum and the mean. Furthermore, we look for the proportion
            of frames that had a prediciton above a given threshold.
            We end up with 4 ndarrays of 4 columns and 1 row (or column vector).
            In the end we just concatenate one after the other and append them
            to our batch. This batch is going to be used as input to the ELM.
        '''
        high_lvl_features = np.empty((len(sequence), 16))
        high_lvl_labels = np.empty((len(sequence), 4))
        for index, (frames, labels) in enumerate(sequence):
          training_pred_emotion_sequence = model.predict(frames)[2]
          max_ = np.amax(training_pred_emotion_sequence, axis=1)
          min_ = np.amin(training_pred_emotion_sequence, axis=1)
          mean = np.mean(training_pred_emotion_sequence, axis=1)
          portion_over_threshold = np.count_nonzero(training_pred_emotion_sequence > 0.25, axis=1)
          portion_over_threshold = portion_over_threshold / max(training_pred_emotion_sequence.shape[1], 1)
          high_lvl_features[index] = np.concatenate((max_, min_, mean, portion_over_threshold), axis=1)
          high_lvl_labels[index] = labels[2]
        return high_lvl_features, high_lvl_labels


    training_high_lvl_features, training_labels = extract_high_level_features(training_sequence)
    test_high_lvl_features, testing_labels = extract_high_level_features(test_sequence)

    sess = tf.Session()
    elm = ELM(sess, *training_high_lvl_features.shape, hidden_num=50, output_len=4)
    elm.feed(training_high_lvl_features, training_labels)
    predictions = elm.test(test_high_lvl_features)
    predictions = np.argmax(predictions, axis=1)
    testing_labels = np.argmax(testing_labels, axis=1)
    cm = confusion_matrix(testing_labels, predictions, labels=np.array([0,1,2,3]))
    with open(args.save_dir + f'/{args.prefix}_elm_confussion_matrix.txt', mode='w') as f:
        f.write(str(cm))
    with open(args.save_dir + f'/{args.prefix}_elm_confussion_matrix.pkl', mode='wb') as f:
        pickle.dump(cm, f, protocol=4)
