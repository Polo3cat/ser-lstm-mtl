#!/bin/bash

python split.py --corpus aibo --out_dir aibo/ 
python split.py --corpus emodb --out_dir emodb/ --acted
python split.py --corpus enterface --out_dir enterface/ --acted
python split.py --corpus iemocap --out_dir iemocap_acted/ --acted
python split.py --corpus iemocap --out_dir iemocap_natural/
python split.py --corpus ldc --out_dir ldc/ --acted
