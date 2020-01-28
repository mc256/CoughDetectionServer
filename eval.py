# %%
import os
import torch
import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch.nn as nn

import pickle
from tqdm import tqdm as tqdm

BATCH_SIZE = 20
INPUT_PIXEL_WIDTH = 8
SEED = 2333333
THRESHOLD = -0.9758867556108074

MODEL = 29
RUN = 'Dec5-1r'

DEVICE = 'cuda:0'
# DEVICE = 'cpu'

DATA_PATH = '/home/jlchen/sandbox/feature/dnn_paper'
LABEL_PATH = '/home/jlchen/sandbox/labels/segmented_fine_labels_w_sliding_win.csv'

# %%
table_fine = pd.read_csv('./fine_grained_annotation.csv', index_col=0)
id_true_fine_list = table_fine[table_fine['label'] == 'Cough']['coarse_grained_annotation_id'].unique()
id_fine_list = table_fine['coarse_grained_annotation_id'].unique()
print('fine Cough coverage count', len(id_true_fine_list))
print('fine coverage count', len(id_fine_list))

# %%
table_coarse = pd.read_csv('./coarse_grained_annotation.csv', index_col=0)
id_true_coarse_list = table_coarse[table_coarse['label'] == True]['id'].unique()
id_coarse_list = table_coarse['id'].unique()
print('coarse Cough coverage count ', len(id_true_coarse_list))
print('coarse coverage count', len(id_coarse_list))

# %%
print(
    "total unique audio samples:",
    len(set(id_coarse_list).union(set(id_fine_list)))
)

print(
    'cough in fine but not in coarse:',
    len(set(id_true_fine_list) - set(id_true_coarse_list))
)

print(
    'cough in coarse but not in fine:',
    len(set(id_true_coarse_list) - set(id_true_fine_list))
)
print(
    "being labelled as cough in coarse-grained set, but not in fine grained set: ",
    len(set(id_true_coarse_list) - set(id_fine_list))
)

# %%
fine_no_cough = set(id_fine_list) - set(id_true_fine_list)
disagree = list(set(id_true_coarse_list).intersection(fine_no_cough))

# %%
segmented_labels = pd.read_csv(LABEL_PATH, index_col=0)

labels_temp, labels_test = train_test_split(
    segmented_labels,
    test_size=0.3,
    random_state=SEED,
    shuffle=False,
)
labels_train, labels_val = train_test_split(
    labels_temp,
    test_size=0.25,
    random_state=SEED,
    shuffle=False,
)

# %%
can_not_use = labels_train['audio'].unique()
coarse = table_coarse[~table_coarse['id'].isin(can_not_use)]
fine = table_fine[~table_fine['coarse_grained_annotation_id'].isin(can_not_use)]


# %%

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


# %%

black_box = ModelC().to(DEVICE).double()
print(black_box)

black_box.load_state_dict(torch.load('./model/cnn_attempt_%s_epoch_%d.pkl' % (RUN, MODEL)))
black_box.eval()

# %%
can_test_cough = set(id_true_fine_list) - set(can_not_use)
can_test_non_cough = set(id_fine_list) - set(id_true_fine_list) - set(can_not_use)
print("cough_examples: ", list(can_test_cough)[:10])
print("non_cough_examples: ", list(can_test_non_cough)[:10])


# %%
def magic(audio_id, active_frame_count=16):
    file_path = os.path.join(DATA_PATH, "dnn2016_%d.pkl" % audio_id)
    with open(file_path, 'rb') as file_handler:
        data = pickle.load(file_handler)

        compact_window = np.array(
            (
                data['zxx_log'],
                np.roll(data['zxx_log'], -1, axis=1),
                np.roll(data['zxx_log'], -2, axis=1),
                np.roll(data['zxx_log'], -3, axis=1),
                np.roll(data['zxx_log'], -4, axis=1),
                np.roll(data['zxx_log'], -5, axis=1),
                np.roll(data['zxx_log'], -6, axis=1),
                np.roll(data['zxx_log'], -7, axis=1),
                np.roll(data['zxx_log'], -8, axis=1),
                np.roll(data['zxx_log'], -9, axis=1),
                np.roll(data['zxx_log'], -10, axis=1),
                np.roll(data['zxx_log'], -11, axis=1),
                np.roll(data['zxx_log'], -12, axis=1),
                np.roll(data['zxx_log'], -13, axis=1),
                np.roll(data['zxx_log'], -14, axis=1),
                np.roll(data['zxx_log'], -15, axis=1),
            )
        )
        compact_window = np.swapaxes(np.swapaxes(np.swapaxes(compact_window, 0, 1), 0, 2), 1, 2).reshape((-1, 64, 16))

        x = th.from_numpy(compact_window).double()
        pred_val = black_box(x.to(DEVICE).view(-1, 1, 64, 16)).to('cpu').detach().numpy()

        prediction = (pred_val[:, 1] - pred_val[:, 0] > THRESHOLD)
        positive = len(prediction[prediction == True]) > active_frame_count
        return positive

    # %%


acc_history = []
for active in range(0, 49):
    wrong = 0
    total = 0
    for sample in tqdm(list(can_test_cough), total=len(list(can_test_cough))):
        try:
            if not magic(sample, active):
                wrong += 1
            total += 1
        except OSError as e:
            pass
        except Exception as e:
            pass

    for sample in tqdm(list(can_test_non_cough), total=len(list(can_test_non_cough))):
        try:
            if magic(sample, active):
                wrong += 1
            total += 1
        except OSError as e:
            pass
        except Exception as e:
            pass

    acc = (total - wrong) / total
    print("accuracy:", acc)
    acc_history.append(acc)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(
    np.arange(0, 49),
    acc_history
)
fig.tight_layout()
fig.show()
fig.savefig('/home/jlchen/sandbox/curve-%s-epoch%d-dec5.png' % (RUN, MODEL))

# %%
