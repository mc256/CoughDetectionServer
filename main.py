import json
import numpy as np
import os
import torch
import pandas as pd
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocess import process_audio

import torch.nn as nn

import pickle
from tqdm import tqdm as tqdm

from http.server import HTTPServer, SimpleHTTPRequestHandler

THRESHOLD = -0.6
DEVICE = 'cuda:0'
#MODEL = 29
MODEL = 49
RUN = 'Dec5-1r'
METHOD = 1


class ModelC(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(9, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(1, 1, kernel_size=(5, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        cnn_out = self.cnn_layers(x)
        flatten = torch.flatten(cnn_out, 1)
        classifier = self.fc_layers(flatten)
        return classifier


black_box = ModelC().to(DEVICE).double()
print(black_box)

black_box.load_state_dict(torch.load('./model/cnn_attempt_%s_epoch_%d.pkl' % (RUN, MODEL)))
black_box.eval()

def run_server():
    class Task_HTTPServer(SimpleHTTPRequestHandler):
        def do_POST(self):
            content_len = int(self.headers.get('content-length', 0))
            post_body = self.rfile.read(content_len)
            audio = np.array(json.loads(post_body))

            features = process_audio(16000, audio)

            compact_window = np.array(
                (
                    features['zxx_log'],
                    np.roll(features['zxx_log'], -1, axis=1),
                    np.roll(features['zxx_log'], -2, axis=1),
                    np.roll(features['zxx_log'], -3, axis=1),
                    np.roll(features['zxx_log'], -4, axis=1),
                    np.roll(features['zxx_log'], -5, axis=1),
                    np.roll(features['zxx_log'], -6, axis=1),
                    np.roll(features['zxx_log'], -7, axis=1),
                    np.roll(features['zxx_log'], -8, axis=1),
                    np.roll(features['zxx_log'], -9, axis=1),
                    np.roll(features['zxx_log'], -10, axis=1),
                    np.roll(features['zxx_log'], -11, axis=1),
                    np.roll(features['zxx_log'], -12, axis=1),
                    np.roll(features['zxx_log'], -13, axis=1),
                    np.roll(features['zxx_log'], -14, axis=1),
                    np.roll(features['zxx_log'], -15, axis=1),
                )
            )
            compact_window = np.swapaxes(np.swapaxes(np.swapaxes(compact_window, 0, 1), 0, 2), 1, 2).reshape(
                (-1, 64, 16))

            x = th.from_numpy(compact_window).double()
            pred_val = black_box(x.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()

            if METHOD == 0:
                prediction = (pred_val[:, 1] - pred_val[:, 0] > THRESHOLD)

                prediction_windowed = np.max(np.array(prediction).reshape((-1, 16)), axis=1)
                prediction_windowed_shift = np.array(
                    (
                        prediction_windowed,
                        np.roll(prediction_windowed, -1),
                        np.roll(prediction_windowed, -2),
                        np.roll(prediction_windowed, -3),
                        np.roll(prediction_windowed, -4),
                        np.roll(prediction_windowed, -5),
                        np.roll(prediction_windowed, -6),
                        np.roll(prediction_windowed, -7),
                        np.roll(prediction_windowed, -8),
                        np.roll(prediction_windowed, -9)
                    )
                )
                largest_window = np.max(np.sum(prediction_windowed_shift, axis=0))


                max = np.interp(largest_window, (0, 9),(-60,10))

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(bytes('{"average":%d,"peak":%d}' % (max, max), 'utf8'))
                print(largest_window, end='')
            else:
                prediction = pred_val[:, 1] - pred_val[:, 0]

                avg = np.average(prediction)
                max = np.max(prediction)

                avg = np.interp(avg, (-4.53,THRESHOLD),(-60,10))
                max = np.interp(max, (-4.53,THRESHOLD),(-60,10))

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(bytes('{"average":%d,"peak":%d}' % (avg, max), 'utf8'))

                print(max)


        def log_message(self, format, *args):
            return

    httpd = HTTPServer(('0.0.0.0', 8088), Task_HTTPServer)
    httpd.serve_forever()
    pass


def main():
    print("server started!")
    run_server()


if __name__ == '__main__':
    main()

