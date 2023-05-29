import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import MobileNetV2
from adabound import AdaBound
import datetime
from pytz import timezone
import pandas as pd
from test import test


TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
KST = timezone("Asia/Seoul")

current_time = datetime.datetime.now(KST).strftime(TIME_FORMAT)


def save_csv(epoch_arr, train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr):
    # make the dataframes with the loss and acc arrays
    df = pd.DataFrame(
        {
            "epoch": epoch_arr,
            "train_loss": train_loss_arr,
            "val_loss": val_loss_arr,
            "train_acc": train_acc_arr,
            "val_acc": val_acc_arr,
        }
    )
    # save the dataframes to csv, and name is current time
    df.to_csv(f"plot/{current_time}.csv")
    return None


def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()


def validate(data_loader, model, criterion):
    model.eval()
    val_acc = 0.0
    total = 0
    val_losses = []

    with torch.no_grad():
        for x, y in data_loader:
            image = x.to(args.device)
            label = y.to(args.device)

            output = model(image)
            label = label.squeeze()

            loss = criterion(output, label)
            val_losses.append(loss.item())

            total += label.size(0)
            val_acc += acc(output, label)

    epoch_val_loss = np.mean(val_losses)
    epoch_val_acc = val_acc / total

    return epoch_val_loss, epoch_val_acc


def train(args, train_loader, val_loader, model):
    train_loss_arr = np.array([])
    val_loss_arr = np.array([])
    train_acc_arr = np.array([])
    val_acc_arr = np.array([])
    epoch_arr = np.array([])

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = AdaBound(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 15  # Number of epochs to wait before early stopping
    early_stop_counter = 0  # Counter for the number of epochs with no improvement

    for epoch in range(args.epochs):
        train_losses = []
        train_acc = 0.0
        total = 0
        print(f"[Epoch {epoch+1} / {args.epochs}]")

        model.train()
        pbar = tqdm(train_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)
            optimizer.zero_grad()

            output = model(image)

            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total

        val_loss, val_acc = validate(val_loader, model, criterion)

        print(f"Epoch {epoch+1}")
        print(f"train_loss: {epoch_train_loss}, val_loss: {val_loss}")
        print(
            f"train_accuracy: {epoch_train_acc*100:.3f}%, val_accuracy: {val_acc*100:.3f}%"
        )
        print(
            f"{epoch_train_loss} {val_loss} {epoch_train_acc*100:.3f} {val_acc*100:.3f}"
        )

        train_loss_arr = np.append(train_loss_arr, epoch_train_loss)
        val_loss_arr = np.append(val_loss_arr, val_loss)
        train_acc_arr = np.append(train_acc_arr, epoch_train_acc)
        val_acc_arr = np.append(val_acc_arr, val_acc)
        epoch_arr = np.append(epoch_arr, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{args.save_path}/{current_time}.pth")

        else:
            early_stop_counter += 1

        save_csv(epoch_arr, train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr)

        if early_stop_counter >= patience:
            print("Early stopping. No improvement in validation loss.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2023 DL Term Project")
    parser.add_argument(
        "--save-path", default="checkpoints/", help="Model's state_dict"
    )
    parser.add_argument("--data", default="data/", type=str, help="data folder")
    parser.add_argument("--mps", action="store_true", help="use mps")
    parser.add_argument(
        "--save_csv", action="store_true", help="save loss and acc to csv"
    )
    parser.add_argument("--test", action="store_true", help="test mode")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mps:
        device = torch.device("mps:0")

    args.device = device

    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """

    # hyperparameters
    args.epochs = 100
    args.learning_rate = 0.01
    args.batch_size = 64

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print("Using Device:", device)
    print("Number of usable GPUs:", torch.cuda.device_count())

    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    print(f"time: {current_time}")
    print("==============================")

    train_loader, val_loader, test_loader = make_data_loader(args)

    model = MobileNetV2()
    model.to(device)

    # Training The Model
    train(args, train_loader, val_loader, model)

    print("=====================================")
    print("Training Finished")
    if args.test:
        pred, true = test(args, test_loader, model)

        accuracy = (true == pred).sum() / len(pred)
        print("Test Accuracy : {:.5f}".format(accuracy))
