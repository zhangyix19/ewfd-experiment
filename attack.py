from re import X
import datahandler as dh
import attacks as wfpattack
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
import time


def calculate_tpr_fpr(y, y_pred, n_classes):
    tprs, fprs = [], []
    for i in range(n_classes):
        # 将类别 i 视为正例，其余类别为负例
        y_i = np.array(y[:, i])
        y_pred_i = np.array(y_pred[:, i])
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_i, y_pred_i, labels=[0, 1]).ravel()
        # 计算 TPR 和 FPR
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)
    return tprs, fprs


def evaluate(y, y_pred, n_classes):
    # 计算宏平均 TPR 和 FPR
    y_pred = np.eye(n_classes)[y_pred.argmax(axis=1)]
    tprs, fprs = calculate_tpr_fpr(y, y_pred, n_classes)
    macro_tpr = np.mean(tprs)
    macro_fpr = np.mean(fprs)
    # 计算微平均 TPR 和 FPR
    tn, fp, fn, tp = confusion_matrix(y.ravel(), y_pred.ravel()).ravel()
    micro_tpr = tp / (tp + fn)
    micro_fpr = fp / (fp + tn)
    # 计算其他指标
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y.argmax(axis=1), y_pred.argmax(axis=1), average="macro", zero_division=0)
    precision = precision_score(
        y.argmax(axis=1), y_pred.argmax(axis=1), average="macro", zero_division=0
    )
    recall = recall_score(y.argmax(axis=1), y_pred.argmax(axis=1), average="macro", zero_division=0)
    return {
        "macro_tpr": macro_tpr,
        "macro_fpr": macro_fpr,
        "micro_tpr": micro_tpr,
        "micro_fpr": micro_fpr,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def writeWriter(writer, eval_result, epoch, mode):
    if writer is None:
        return
    for key in eval_result:
        writer.add_scalar(f"{mode}/{key}", eval_result[key], epoch)


def load_array(features, labels, batch_size, is_train=True):
    if len(features) < batch_size:
        return range(0)
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train)


def df(
    length,
    train_features,
    train_labels,
    valid_features,
    valid_labels,
    test_features,
    test_labels,
    writer=None,
    num_epoch=1000,
    usesize=False,
    gpu="0",
    tik_tok=False,
    savedir="data/run",
    testonly=False,
    loaddir="data/run",
):
    name = "tiktok" if tik_tok else "df"
    num_epoch = 0 if testonly else num_epoch
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    learning_rate = 0.002
    length_for_df = {
        5000: 18,
        10000: 37,
        15000: 57,
        20000: 76,
    }
    if tik_tok:
        train_features = dh.utils.get_tiktok_features(train_features)
        test_features = dh.utils.get_tiktok_features(test_features)
        valid_features = dh.utils.get_tiktok_features(valid_features)
    else:
        train_features = dh.utils.get_df_features(train_features, usesize)
        valid_features = dh.utils.get_df_features(valid_features, usesize)
        test_features = dh.utils.get_df_features(test_features, usesize)
    num_classes = max(np.max(train_labels), np.max(valid_labels), np.max(test_labels)) + 1
    batch_size = min(num_classes * 8, 256)
    if train_features.shape[0] > 20000:
        batch_size = 400
    # to one hot
    train_labels = np.eye(num_classes)[train_labels]
    valid_labels = np.eye(num_classes)[valid_labels]
    test_labels = np.eye(num_classes)[test_labels]
    # to tensor
    train_features = torch.tensor(train_features.reshape(-1, 1, length), dtype=torch.float32)
    valid_features = torch.tensor(valid_features.reshape(-1, 1, length), dtype=torch.float32)
    test_features = torch.tensor(test_features.reshape(-1, 1, length), dtype=torch.float32)

    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    valid_labels = torch.tensor(valid_labels, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # to dataloader
    train_iter = load_array(train_features, train_labels, batch_size, is_train=True)
    valid_iter = load_array(valid_features, valid_labels, batch_size, is_train=False)
    test_iter = load_array(test_features, test_labels, batch_size, is_train=False)

    # set model
    net = (
        wfpattack.df.DF(num_classes, length_for_df[length])
        if not testonly
        else torch.load(os.path.join(loaddir, f"{name}.pt"))
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )
    net = net.to(device)

    # for each epoch
    loss_min, patience = 9999999999, 0
    for epoch in tqdm(range(num_epoch)):
        # train
        net.train()
        train_sum_loss = 0.0
        count = 0
        y_true = np.empty(shape=(0, num_classes))
        y_pred = np.empty(shape=(0, num_classes))
        for index, data in enumerate(train_iter):
            X, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outs = net(X)
            loss = criterion(outs, y)
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.data
            count += outs.shape[0]
            y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

        eval_result = evaluate(y_true, y_pred, num_classes)
        eval_result["loss"] = train_sum_loss / count
        writeWriter(writer, eval_result, epoch, "train")
        # evaluate on valid
        net.eval()
        with torch.no_grad():
            valid_sum_loss = 0.0
            count = 0
            y_true = np.empty(shape=(0, num_classes))
            y_pred = np.empty(shape=(0, num_classes))
            for index, data in enumerate(valid_iter):
                X, y = data[0].to(device), data[1].to(device)
                outs = net(X)
                loss = criterion(outs, y)

                valid_sum_loss += loss.data
                count += outs.shape[0]
                y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)
            eval_result = evaluate(y_true, y_pred, num_classes)
            eval_result["loss"] = valid_sum_loss / count
            writeWriter(writer, eval_result, epoch, "valid")
            if valid_sum_loss / count < loss_min:
                loss_min = valid_sum_loss / count
                patience = 0
            else:
                patience += 1

        # evaluate on test
        with torch.no_grad():
            count = 0
            y_true = np.empty(shape=(0, num_classes))
            y_pred = np.empty(shape=(0, num_classes))
            for index, data in enumerate(test_iter):
                X, y = data[0].to(device), data[1].to(device)
                outs = net(X)
                count += outs.shape[0]
                y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)
            eval_result = evaluate(y_true, y_pred, num_classes)
            writeWriter(writer, eval_result, epoch, "test")

        # save model every 50 epoch
        if epoch % 50 == 0:
            os.makedirs(os.path.join(savedir, name), exist_ok=True)
            torch.save(net, os.path.join(savedir, name, f"{name}_{epoch}.pt"))

        # stop when validloss not change for 50 epoch
        if patience > 50 and epoch > num_epoch / 2:
            print("early stop")
            break
        # evaluate on test
        net.eval()

    # save model
    if not testonly:
        model_file = f"{name}.pt"
        os.makedirs(os.path.join(savedir), exist_ok=True)
        if os.path.exists(os.path.join(savedir, model_file)):
            os.rename(
                os.path.join(savedir, model_file),
                os.path.join(savedir, str(int(time.time()))) + model_file,
            )
        torch.save(net, os.path.join(savedir, model_file))

    # final test
    with torch.no_grad():
        count = 0
        y_true = np.empty(shape=(0, num_classes))
        y_pred = np.empty(shape=(0, num_classes))
        for index, data in enumerate(test_iter):
            X, y = data[0].to(device), data[1].to(device)
            outs = net(X)
            count += outs.shape[0]
            y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)
        final_result = evaluate(y_true, y_pred, num_classes)
    del net
    return final_result


def kfp(length, train_features, train_labels, test_features, test_labels):
    num_epoch = args.epoch if args.epoch > 0 else 10
    train_features = dh.utils.get_kfp_features(train_features)
    test_features = dh.utils.get_kfp_features(test_features)
    # regenrate features
    features = np.concatenate((train_features, test_features), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    num_classes = int(np.max(labels) + 1)
    for epoch in tqdm(range(num_epoch)):
        randn = random.randint(1, 500)
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=randn
        )
        train_features, test_features = wfpattack.kfp.extract_dill(train_features, test_features)
        y_true = test_labels
        y_pred = wfpattack.kfp.RF_openworld(
            train_features, test_features, train_labels, test_labels
        )
        y_true = np.eye(num_classes)[y_true]
        y_pred = np.eye(num_classes)[y_pred]
        eval_result = evaluate(y_true, y_pred, num_classes)
        writeWriter(eval_result, epoch, "test")


def cumul(length, train_features, train_labels, test_features, test_labels):
    num_epoch = args.epoch if args.epoch > 0 else 10
    train_features = dh.utils.get_cumul_features(train_features)
    test_features = dh.utils.get_cumul_features(test_features)
    # regenrate features
    features = np.concatenate((train_features, test_features), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    num_classes = int(np.max(labels) + 1)
    for epoch in tqdm(range(num_epoch)):
        randn = random.randint(1, 500)
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=randn
        )
        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)
        y_true = test_labels
        y_pred = wfpattack.cumul.train(train_features, train_labels, test_features, test_labels)
        y_true = np.eye(num_classes)[y_true]
        y_pred = np.eye(num_classes)[y_pred]
        eval_result = evaluate(y_true, y_pred, num_classes)
        writeWriter(eval_result, epoch, "test")


def awf_cnn(length, train_features, train_labels, test_features, test_labels):
    learning_rate = 0.001
    num_epoch = args.epoch
    train_features = dh.utils.get_awf_features(train_features, args.http)
    test_features = dh.utils.get_awf_features(test_features, args.http)

    num_classes = int(np.max(np.append(train_labels, test_labels)) + 1)
    print(num_classes)
    batch_size = num_classes * 4
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    train_features = torch.tensor(train_features.reshape(-1, 1, length), dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_features = torch.tensor(test_features.reshape(-1, 1, length), dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    train_iter = load_array(train_features, train_labels, batch_size)
    test_iter = load_array(test_features, test_labels, batch_size, False)

    net = wfpattack.awf.AWF_CNN(num_classes)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    net = net.to(device)
    # train
    net.train()
    for epoch in tqdm(range(num_epoch)):
        train_sum_loss = 0.0
        count = 0
        y_true = np.empty(shape=(0, num_classes))
        y_pred = np.empty(shape=(0, num_classes))

        for index, train_data in enumerate(train_iter):
            train_features, y = train_data[0].to(device), train_data[1].to(device)
            optimizer.zero_grad()
            outs = net(train_features)

            loss = criterion(outs, y.argmax(1))
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.data
            count += outs.shape[0]
            y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

        eval_result = evaluate(y_true, y_pred, num_classes)
        writeWriter(eval_result, epoch, "train")

        net.eval()
        with torch.no_grad():
            valid_sum_loss = 0.0
            count = 0
            y_true = np.empty(shape=(0, num_classes))
            y_pred = np.empty(shape=(0, num_classes))
            for index, valid_data in enumerate(test_iter):
                train_features, y = valid_data[0].to(device), valid_data[1].to(device)
                outs = net(train_features)
                loss = criterion(outs, y.argmax(1))

                valid_sum_loss += loss.data
                count += outs.shape[0]
                y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

            eval_result = evaluate(y_true, y_pred, num_classes)
            writeWriter(eval_result, epoch, "valid")
        net.train()
    del net


def awf_lstm(length, train_features, train_labels, test_features, test_labels):
    learning_rate = 0.001
    num_epoch = args.epoch
    train_features = dh.utils.get_awf_features(train_features, args.http, "lstm")
    test_features = dh.utils.get_awf_features(test_features, args.http, "lstm")
    print(np.max(train_features.reshape(-1)))
    num_classes = int(np.max(np.append(train_labels, test_labels)) + 1)
    print(num_classes)
    batch_size = num_classes * 4
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    train_iter = load_array(train_features, train_labels, batch_size)
    test_iter = load_array(test_features, test_labels, batch_size, False)

    net = wfpattack.awf.AWF_LSTM(num_classes, batch_size)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    net = net.to(device)
    # train
    net.train()
    for epoch in tqdm(range(num_epoch)):
        train_sum_loss = 0.0
        count = 0
        y_true = np.empty(shape=(0, num_classes))
        y_pred = np.empty(shape=(0, num_classes))

        for index, train_data in enumerate(train_iter):
            train_features, y = train_data[0].to(device), train_data[1].to(device)
            optimizer.zero_grad()
            outs = net(train_features)
            loss = criterion(outs, y.argmax(1))
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.data
            count += outs.shape[0]
            y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

        eval_result = evaluate(y_true, y_pred, num_classes)
        writeWriter(eval_result, epoch, "train")

        net.eval()
        with torch.no_grad():
            valid_sum_loss = 0.0
            count = 0
            y_true = np.empty(shape=(0, num_classes))
            y_pred = np.empty(shape=(0, num_classes))
            for index, valid_data in enumerate(test_iter):
                train_features, y = valid_data[0].to(device), valid_data[1].to(device)
                outs = net(train_features)
                loss = criterion(outs, y.argmax(1))

                valid_sum_loss += loss.data
                count += outs.shape[0]
                y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

            eval_result = evaluate(y_true, y_pred, num_classes)
            writeWriter(eval_result, epoch, "valid")

        net.train()
    del net


def awf_sdae(length, train_features, train_labels, test_features, test_labels):
    learning_rate = 0.001
    num_epoch = args.epoch
    train_features = dh.utils.get_awf_features(train_features, args.http)
    test_features = dh.utils.get_awf_features(test_features, args.http)

    num_classes = int(np.max(np.append(train_labels, test_labels)) + 1)
    print(num_classes)
    batch_size = num_classes * 4
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    train_iter = load_array(train_features, train_labels, batch_size)
    test_iter = load_array(test_features, test_labels, batch_size, False)

    net = wfpattack.awf.AWF_SDAE(num_classes)
    for n, p in list(net.named_parameters()):
        if "gamma" not in n and "beta" not in n:
            p.data.normal_(0, 0.02)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    net = net.to(device)
    # train
    net.train()
    for epoch in tqdm(range(num_epoch)):
        train_sum_loss = 0.0
        count = 0
        y_true = np.empty(shape=(0, num_classes))
        y_pred = np.empty(shape=(0, num_classes))

        for index, train_data in enumerate(train_iter):
            train_features, y = train_data[0].to(device), train_data[1].to(device)
            optimizer.zero_grad()
            outs = net(train_features)
            loss = criterion(outs, y.argmax(1))
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.data
            count += outs.shape[0]
            y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
            y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

        eval_result = evaluate(y_true, y_pred, num_classes)
        writeWriter(eval_result, epoch, "train")

        net.eval()
        with torch.no_grad():
            valid_sum_loss = 0.0
            count = 0
            y_true = np.empty(shape=(0, num_classes))
            y_pred = np.empty(shape=(0, num_classes))
            for index, valid_data in enumerate(test_iter):
                train_features, y = valid_data[0].to(device), valid_data[1].to(device)
                outs = net(train_features)
                loss = criterion(outs, y.argmax(1))

                valid_sum_loss += loss.data
                count += outs.shape[0]
                y_true = np.concatenate((y_true, y.cpu().detach().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, outs.cpu().detach().numpy()), axis=0)

            eval_result = evaluate(y_true, y_pred, num_classes)
            writeWriter(eval_result, epoch, "valid")
        net.train()
    del net


def run_attack(
    attack,
    length,
    train_features,
    train_labels,
    valid_features,
    valid_labels,
    test_features,
    test_labels,
    num_epoch=1000,
    writer=None,
    gpu="0",
    save=False,
    savedir="data/run",
    testonly=False,
    loaddir="",
):
    if attack == "df":
        return df(
            length,
            train_features,
            train_labels,
            valid_features,
            valid_labels,
            test_features,
            test_labels,
            num_epoch=num_epoch,
            writer=writer,
            savedir=savedir,
            gpu=gpu,
            tik_tok=False,
            testonly=testonly,
            loaddir=loaddir,
        )
    elif attack == "tiktok":
        return df(
            length,
            train_features,
            train_labels,
            valid_features,
            valid_labels,
            test_features,
            test_labels,
            num_epoch=num_epoch,
            writer=writer,
            savedir=savedir,
            gpu=gpu,
            tik_tok=True,
            testonly=testonly,
            loaddir=loaddir,
        )
    elif attack == "kfp":
        kfp(length, train_features, train_labels, test_features, test_labels)
    elif attack == "cumul":
        cumul(length, train_features, train_labels, test_features, test_labels)
    elif attack == "cnn":
        awf_cnn(length, train_features, train_labels, test_features, test_labels)
    elif attack == "lstm":
        awf_lstm(length, train_features, train_labels, test_features, test_labels)
    elif attack == "sdae":
        awf_sdae(length, train_features, train_labels, test_features, test_labels)
    else:
        raise Exception(f"attack {attack} not implemented")
