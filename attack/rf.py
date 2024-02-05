import math

import numpy as np
import torch
import torch.nn as nn

from .base import DNNAttack
import multiprocessing as mp
from tqdm import tqdm
from torch.optim import lr_scheduler
import psutil
import time


def make_first_layers(in_channels=1, out_channel=32):
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [
        conv2d1,
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [
        conv2d2,
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)


def make_layers(cfg, in_channels=32):
    layers = []

    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)


class RF_model(nn.Module):
    def __init__(self, features, num_classes=95, init_weights=True):
        super(RF_model, self).__init__()
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        self.first_layer = make_first_layers()
        self.features = features
        self.class_num = num_classes
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {"N": [128, 128, "M", 256, 256, "M", 512]}


class RF(DNNAttack):
    name = "RF"

    def __init__(self, trace_length, num_classes, gpu):
        super().__init__(trace_length, num_classes, gpu)
        self.model = None
        self.num_epochs = 600
        self.batch_size = 800
        self.learning_rate = 0.0005
        self.criterion = nn.CrossEntropyLoss()

    def init_model(self, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        self.model = RF_model(
            make_layers(cfg["N"] + [self.num_classes]),
            num_classes=self.num_classes,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.001,
        )

    def on_train_epoch(self, epoch):
        lr = self.learning_rate * (0.2 ** (epoch / self.num_epochs))
        for para_group in self.optimizer.param_groups:
            para_group["lr"] = lr

    @staticmethod
    def packets_per_slot(trace):
        time = trace[:, 0]
        direction = trace[:, 1]
        # Length of TAM
        max_matrix_len = 1800
        # Maximum Load Time
        maximum_load_time = 80

        feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
        for i in range(0, len(direction)):
            if direction[i] > 0:
                if time[i] >= maximum_load_time:
                    feature[0][-1] += 1
                else:
                    idx = int(time[i] * (max_matrix_len - 1) / maximum_load_time)
                    feature[0][idx] += 1
            if direction[i] < 0:
                if time[i] >= maximum_load_time:
                    feature[1][-1] += 1
                else:
                    idx = int(time[i] * (max_matrix_len - 1) / maximum_load_time)
                    feature[1][idx] += 1

        return feature

    def data_preprocess(self, traces, labels):
        # make traces to have same trace_length
        # traces = [np.array(trace[: self.trace_length]) for trace in traces]
        # traces = [
        #     np.pad(trace, ((0, self.trace_length - trace.shape[0]), (0, 0)), "constant")
        #     if trace.shape[0] < self.trace_length
        #     else trace
        #     for trace in traces
        # ]
        for trace in traces:
            trace[:, 0] = trace[:, 0] - trace[0, 0]
        # get features
        # traces = [self.packets_per_slot(trace) for trace in traces]

        while True:
            cpu_num = int((psutil.cpu_count() - psutil.cpu_percent()) * 0.7)
            if cpu_num > 10:
                break
            time.sleep(5)

        with mp.Pool(cpu_num) as pool:
            tqdm_iter = tqdm(
                pool.imap(self.packets_per_slot, traces),
                total=len(traces),
                desc="Preprocessing RF features",
            )
            traces = list(tqdm_iter)
        traces = np.array(traces)
        traces = torch.unsqueeze(torch.from_numpy(traces), dim=1).type(torch.FloatTensor)
        traces = traces.view(traces.size(0), 1, 2, -1)

        labels = torch.tensor(labels, dtype=torch.long)
        labels = torch.eye(self.num_classes)[labels]
        return {"traces": traces, "labels": labels}
