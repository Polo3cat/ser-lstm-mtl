#!/bin/bash
# args -> learning_rate gender_weight acted_weight private_layers prefix
args='0.001 0 0 0 base'

python3.7 hpc_trainer.py ./leave_speaker_out/aibo/train_ids.txt ./leave_speaker_out/aibo/validation_ids.txt ./leave_speaker_out/aibo/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/aibo/ $args
python3.7 hpc_trainer.py ./leave_speaker_out/enterface/train_ids.txt ./leave_speaker_out/enterface/validation_ids.txt ./leave_speaker_out/enterface/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/enterface/ $args
python3.7 hpc_trainer.py ./leave_speaker_out/emodb/train_ids.txt ./leave_speaker_out/emodb/validation_ids.txt ./leave_speaker_out/emodb/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/emodb/ $args

python3.7 hpc_trainer.py ./leave_corpus_out/aibo/train_ids.txt ./leave_corpus_out/aibo/validation_ids.txt ./leave_corpus_out/aibo/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/aibo/ $args
python3.7 hpc_trainer.py ./leave_corpus_out/enterface/train_ids.txt ./leave_corpus_out/enterface/validation_ids.txt ./leave_corpus_out/enterface/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/enterface/ $args
python3.7 hpc_trainer.py ./leave_corpus_out/emodb/train_ids.txt ./leave_corpus_out/emodb/validation_ids.txt ./leave_corpus_out/emodb/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/emodb/ $args
