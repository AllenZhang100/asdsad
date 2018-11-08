import argparse, pickle
import numpy as np
import os

import torch
from pytux import Tux
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import *
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

levels = [
    '01 - Welcome to Antarctica.stl',
    '02 - The Journey Begins.stl',
    '03 - Via Nostalgica.stl',
    '04 - Tobgle Road.stl',
    '05 - The Somewhat Smaller Bath.stl',
]


def get_action(logits):
    probs = 1. / (1. + np.exp(-logits.detach().numpy()))
    bits = np.array([np.random.uniform()<=p for p in probs]).astype(int)
    return int(np.packbits(bits)[0] >> 2)

def test(iterations, H=500):
    # Load the model
    model = Model()
    model.load_state_dict(torch.load(os.path.join(dirname, 'model.th'), map_location=lambda storage, loc: storage))
    model.eval()

    positions = {level : [] for level in levels}

    for it in range(iterations):
        for level in levels:
            p = 0.0
            print (level)
            T = Tux('data/levels/world1/%s'%level, 128, 128, acting=True, visible=True, synchronized=True)

            # Restart Tux
            T.restart()
            if not T.waitRunning():
                exit(0)

            fid, act, state, obs = T.step(0)
            policy = model.policy()

            for t in range(H):
                
                tux_mask = (obs['label'] % 8) == 4

                xs, ys = np.argwhere(tux_mask).T
                try:
                    x = int(xs.mean())
                    y = int(ys.mean())
                except:
                    x = 64
                    y = 64

                img = obs['image']
                img = np.pad(img, ((32,32), (32,32), (0,0)), mode='constant', constant_values=127)
                img = img[x:x+64,y:y+64]

                img = (img - [4.417,3.339,4.250]) / [23.115,19.552,20.183]
                img = torch.as_tensor(img).float()
                logits = policy(img.permute(2,0,1))
                a = get_action(logits)
                try:
                    fid, act, state, obs = T.step(a)
                except TypeError as e:
                    print (e)
                    break
                if state['is_dying']:
                    break
                p = max(p, state['position'])
            positions[level].append(p)


    for level, scores in positions.items():
        print ("level: %s, mean = %.3f, std = %.3f"%(level,np.mean(scores),np.std(scores)))



if __name__ == '__main__':
    test(iterations=10, H=2000)