#!/bin/bash

python_bin=$HOME/optevo/env/bin/python

$python_bin main.py \
  --seed=3 \
  --out_dir=$HOME/optevo/results \
  --data_dir=$HOME/optevo/data \
  --arch="ff" \
  --max_gen=5000 \
  --pop_size=50 \
  --elites=1 \
  --crossover_p=0.5 \
  --tournament_size=10 \
  --noise_p=0.001 \
  --noise_sigma=0.01 \
  --criterion="batch_acc" \
  --batch_size=128 
