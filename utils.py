#!/usr/bin/env python 

# Copyright 2021 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import numpy as np 
import pickle

def read_result_file(file_path:str): 
    data = pickle.load(open(file_path, 'rb'))
    threshold = data['run_0']['threshold']
    errs_normal = data['run_0']['unseennormal_err']
    errs_attack = data['run_0']['attacks_list_err']
    errs_train =  data['run_0']['trainingerrors']
    return threshold, errs_attack, errs_normal, errs_train

def label_window(yhat, window_size:int=3): 
    N = len(yhat)
    threshold = np.ceil(window_size/2)
    yhat_hat = []
    score_hat = []

    for i in range(0, N-window_size): 
        win = yhat[i:i+window_size]
        score_hat.append(win.mean())
        if win.sum() >= threshold: 
            yhat_hat.append(1)
        else: 
            yhat_hat.append(0)
    
    yhat_hat = np.array(yhat_hat)
    score_hat = np.array(score_hat)
    return yhat_hat, score_hat

def get_rates(ys, yhats, verbose:bool=False):
    tp, tn, fp, fn = 0., 0., 0., 0.
    for y, yhat in zip(ys, yhats): 
        if  y == 1 and yhat == 1: 
            tp += 1
        elif y == 0 and yhat == 0: 
            tn += 1
        elif y == 1 and yhat == 0: 
            fn += 1
        else: 
            fp += 1
    
    acc = (tp+tn)/(tp+tn+fp+fn)
    prec = tp/(tp+fp)
    prev = (tp+fn)/(tp+tn+fp+fn)
    reca = tp/(tp+fn)
    fm = 2*prec*reca/(prec+reca)
    if verbose: 
        print(''.join(['Accuracy: ', str(acc)]))
        print(''.join(['Precision: ', str(prec)]))
        print(''.join(['Recall: ', str(reca)]))
        print(''.join(['F-score: ', str(fm)]))
        print(''.join(['Prevelance: ', str(prev)]))


    return acc, prec, prev, reca, fm 