from .base import DNNAttack
import torch.nn as nn
import numpy as np
import torch


class DF_model(nn.Module):
    def __init__(self, num_classes=2, length=18):
        super(DF_model, self).__init__()

        filter_num = ["None", 32, 64, 128, 256]
        kernel_size = ["None", 8, 8, 8, 8]
        conv_stride_size = ["None", 1, 1, 1, 1]
        pool_stride_size = ["None", 4, 4, 4, 4]
        pool_size = ["None", 8, 8, 8, 8]
        pool_padding = 0
        length_after_extraction = length

        self.feature_extraction = nn.Sequential(
            # block1
            nn.Conv1d(
                in_channels=1,
                out_channels=filter_num[1],
                kernel_size=kernel_size[1],
                stride=conv_stride_size[1],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv1d(
                in_channels=filter_num[1],
                out_channels=filter_num[1],
                kernel_size=kernel_size[1],
                stride=conv_stride_size[1],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_padding
            ),
            nn.Dropout(p=0.1),
            # block2
            nn.Conv1d(
                in_channels=filter_num[1],
                out_channels=filter_num[2],
                kernel_size=kernel_size[2],
                stride=conv_stride_size[2],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=filter_num[2],
                out_channels=filter_num[2],
                kernel_size=kernel_size[2],
                stride=conv_stride_size[2],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_padding
            ),
            nn.Dropout(p=0.1),
            # block3
            nn.Conv1d(
                in_channels=filter_num[2],
                out_channels=filter_num[3],
                kernel_size=kernel_size[3],
                stride=conv_stride_size[3],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=filter_num[3],
                out_channels=filter_num[3],
                kernel_size=kernel_size[3],
                stride=conv_stride_size[3],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_padding
            ),
            nn.Dropout(p=0.1),
            # block4
            nn.Conv1d(
                in_channels=filter_num[3],
                out_channels=filter_num[4],
                kernel_size=kernel_size[4],
                stride=conv_stride_size[4],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=filter_num[4],
                out_channels=filter_num[4],
                kernel_size=kernel_size[4],
                stride=conv_stride_size[4],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_padding
            ),
            nn.Dropout(p=0.1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=filter_num[4] * length_after_extraction, out_features=512, bias=False
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=512, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x)
        return x


class DF(DNNAttack):
    name = "DF"
    length_for_df = {
        5000: 18,
        10000: 37,
        15000: 57,
        20000: 76,
    }

    def __init__(self, trace_length, num_classes, gpu, n_jobs=10):
        super().__init__(trace_length, num_classes, gpu)
        self.model = None
        self.num_epochs = 1000
        self.batch_size = 400
        self.learning_rate = 0.002
        self.criterion = nn.CrossEntropyLoss()

    def init_model(self, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        self.model = DF_model(
            num_classes=self.num_classes, length=self.length_for_df[self.trace_length]
        )
        self.optimizer = torch.optim.Adamax(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

    def data_preprocess(self, traces, labels):
        traces = [np.array(trace[: self.trace_length]) for trace in traces]
        traces = [
            (
                np.pad(trace, ((0, self.trace_length - trace.shape[0]), (0, 0)), "constant")
                if trace.shape[0] < self.trace_length
                else trace
            )
            for trace in traces
        ]
        traces = [np.sign(trace[:, 1]).reshape(1, -1) for trace in traces]
        traces = torch.tensor(np.array(traces), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        labels = torch.eye(self.num_classes)[labels]
        return {"traces": traces, "labels": labels}
