#!/bin/bash

python3.7 hpc_trainer.py ./leave_speaker_out/iemocap_natural/train_ids.txt ./leave_speaker_out/iemocap_natural/validation_ids.txt ./leave_speaker_out/iemocap_natural/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/iemocap_natural/
python3.7 hpc_trainer.py ./leave_speaker_out/ldc/train_ids.txt ./leave_speaker_out/ldc/validation_ids.txt ./leave_speaker_out/ldc/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/ldc/
python3.7 hpc_trainer.py ./leave_speaker_out/iemocap_acted/train_ids.txt ./leave_speaker_out/iemocap_acted/validation_ids.txt ./leave_speaker_out/iemocap_acted/test_ids.txt ./frames_labels.pkl ./leave_speaker_out/iemocap_acted/

python3.7 hpc_trainer.py ./leave_corpus_out/iemocap_natural/train_ids.txt ./leave_corpus_out/iemocap_natural/validation_ids.txt ./leave_corpus_out/iemocap_natural/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/iemocap_natural/
python3.7 hpc_trainer.py ./leave_corpus_out/ldc/train_ids.txt ./leave_corpus_out/ldc/validation_ids.txt ./leave_corpus_out/ldc/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/ldc/
python3.7 hpc_trainer.py ./leave_corpus_out/iemocap_acted/train_ids.txt ./leave_corpus_out/iemocap_acted/validation_ids.txt ./leave_corpus_out/iemocap_acted/test_ids.txt ./frames_labels.pkl ./leave_corpus_out/iemocap_acted/
