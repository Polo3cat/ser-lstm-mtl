#!/bin/bash

python3.7 hpc_trainer.py ./leave_speaker_out/aibo/train_ids.txt ./leave_speaker_out/aibo/validation_ids.txt ./leave_speaker_out/aibo/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/aibo/
python3.7 hpc_trainer.py ./leave_speaker_out/enterface/train_ids.txt ./leave_speaker_out/enterface/validation_ids.txt ./leave_speaker_out/enterface/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/enterface/
python3.7 hpc_trainer.py ./leave_speaker_out/emodb/train_ids.txt ./leave_speaker_out/emodb/validation_ids.txt ./leave_speaker_out/emodb/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/emodb/

python3.7 hpc_trainer.py ./leave_corpus_out/aibo/train_ids.txt ./leave_corpus_out/aibo/validation_ids.txt ./leave_corpus_out/aibo/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/aibo/
python3.7 hpc_trainer.py ./leave_corpus_out/enterface/train_ids.txt ./leave_corpus_out/enterface/validation_ids.txt ./leave_corpus_out/enterface/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/enterface/
python3.7 hpc_trainer.py ./leave_corpus_out/emodb/train_ids.txt ./leave_corpus_out/emodb/validation_ids.txt ./leave_corpus_out/emodb/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/emodb/
