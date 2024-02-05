import os
from os.path import join

from . import metrics
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Attack:
    def __init__(self, trace_length):
        self.trace_length = trace_length

    def data_preprocess(self, traces, labels):
        raise NotImplementedError

    def run_attack(self):
        raise NotImplementedError


class DNNAttack(Attack):
    name = "DNNAttack"

    def __init__(self, trace_length, num_classes, gpu):
        super().__init__(trace_length)
        self.num_classes = num_classes
        self.device = torch.device(
            f"cuda:{gpu}" if gpu != "cpu" and torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.num_epochs = None
        self.batch_size = None
        self.learning_rate = None

    def init_model(self, learning_rate):
        raise NotImplementedError

    @staticmethod
    def load_array(features, labels, batch_size, is_train=True):
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train)

    @staticmethod
    def writeWriter(writer, eval_result, epoch, notes):
        if writer:
            for key, value in eval_result.items():
                writer.add_scalar(f"{notes}/{key}", value, epoch)

    def on_train_epoch(self, epoch):
        pass

    def train(
        self,
        features_train,
        labels_train,
        features_valid,
        labels_valid,
        writer=None,
        num_epochs=0,
        batch_size=0,
        learning_rate=0,
        save_root=None,
        test={},
    ):
        num_epochs = num_epochs or self.num_epochs
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        assert (
            num_epochs and batch_size and learning_rate
        ), f"num_epochs : {num_epochs}, batch_size : {batch_size}, learning_rate : {learning_rate}"
        self.init_model(learning_rate)
        device = self.device
        net = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        train_iter = self.load_array(features_train, labels_train, batch_size, is_train=True)

        net.to(device)
        earlystop_shreshold, earlystop_min_epoch = num_epochs // 10, num_epochs // 3
        loss_min, patience = np.inf, 0
        for epoch in tqdm(range(num_epochs), desc="training"):
            self.on_train_epoch(epoch)
            self.model.train()
            train_sum_loss = 0.0
            y_true, y_pred = [], []
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outs = net(X)
                loss = criterion(outs, y)
                loss.backward()
                optimizer.step()

                train_sum_loss += loss.item()
                y_true.append(y.cpu().detach().numpy())
                y_pred.append(outs.cpu().detach().numpy())
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            metrics_dict = metrics.cal_metrics(y_true, y_pred, self.num_classes)
            metrics_dict.update({"loss": train_sum_loss / y_true.shape[0]})

            self.writeWriter(writer, metrics_dict, epoch, "train")
            if save_root and epoch % 50 == 0:
                self.save_model(save_root, epoch)
            # test
            if test:
                for test_name, test_data in test.items():
                    test_result = self.evaluate(
                        test_data["features"],
                        test_data["labels"],
                        writer,
                        batch_size,
                    )
                    self.writeWriter(writer, test_result, epoch, "test-" + test_name)
            # validation
            valid_result = self.evaluate(features_valid, labels_valid, batch_size=batch_size)
            self.writeWriter(writer, valid_result, epoch, "valid")

            if valid_result["loss"] < loss_min:
                loss_min = valid_result["loss"]
                patience = 0
            if patience >= earlystop_shreshold and epoch >= earlystop_min_epoch:
                break
            patience += 1
        if save_root:
            self.save_model(save_root)

    def save_model(self, save_root, epoch=None):
        model_dir = join(save_root, self.name)
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"model_{epoch}.pkl" if epoch else "model.pkl"
        torch.save(self.model, join(model_dir, model_name))

    def load_model(self, save_root, epoch=None):
        model_dir = join(save_root, self.name)
        model_name = f"model_{epoch}.pkl" if epoch else "model.pkl"
        assert os.path.exists(join(model_dir, model_name)), f"model not found in {model_dir}"
        self.model = torch.load(join(model_dir, model_name), map_location=self.device)

    def evaluate(self, features, labels, writer=None, batch_size=None, load_dir=None, epoch=None):
        batch_size = batch_size or self.batch_size
        assert batch_size, f"batch_size : {batch_size}"
        if load_dir:
            self.load_model(load_dir, epoch)
        assert self.model and self.criterion, "init model with criterion first"
        device = self.device
        net = self.model
        try:
            eval_iter = self.load_array(features, labels, batch_size, is_train=False)
        except:
            print("load array error, maybe use tensor instead")
            return
        net.to(device)
        net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            sum_loss = 0.0
            for X, y in eval_iter:
                X, y = X.to(device), y.to(device)
                outs = net(X)
                loss = self.criterion(outs, y)
                sum_loss += loss.item()
                y_true.append(y.cpu().detach().numpy())
                y_pred.append(outs.cpu().detach().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        metrics_dict = metrics.cal_metrics(y_true, y_pred, self.num_classes)
        metrics_dict.update({"loss": sum_loss / y_true.shape[0]})

        return metrics_dict
