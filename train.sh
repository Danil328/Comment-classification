#!/usr/bin/env bash
python3 preprocess\ data.py
python3 train_model.py --action train
python3 train_model.py --action val
python3 train_model.py --action test