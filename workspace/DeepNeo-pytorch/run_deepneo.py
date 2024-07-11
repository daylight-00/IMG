#!/usr/bin/python
import os
import sys

mhc_class = sys.argv[1]
predtype = sys.argv[2]
Inputname = sys.argv[3]
Resultname = sys.argv[4]

def run_cnn(model_path, input_name, result_name, device='cuda'):
    os.system(f'python cnn.py {model_path} {input_name} {result_name} --device {device}')

if mhc_class == "class1" and predtype == 'tcr':
    run_cnn('data/tcr1-pan.pkl.gz', Inputname, Resultname, device='cuda:3')
    print("\nThe running is completed!\n")

if mhc_class == "class1" and predtype == 'mhc':
    run_cnn('data/mhc1-pan.pkl.gz', Inputname, Resultname, device='cuda:3')
    print("\nThe running is completed!\n")

if mhc_class == "class2" and predtype == 'mhc':
    run_cnn('data/mhc2-pan.pkl.gz', Inputname, Resultname, device='cuda:0')
    print("\nThe running is completed!\n")

if mhc_class == "class2" and predtype == 'tcr':
    run_cnn('data/tcr2-pan.pkl.gz', Inputname, Resultname, device='cuda:0')
    print("\nThe running is completed!\n")
