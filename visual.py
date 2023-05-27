import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel
from adabound import AdaBound
import datetime
from pytz import timezone
import pandas as pd
import os
import matplotlib.pyplot as plt


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_PATH, "plot")


# get csv file and create dataframe
def get_csv_file():
    csv_file = os.listdir(CSV_PATH)
    csv_file = sorted(csv_file)
    return csv_file[-1]


def read_csv_file(file):
    return pd.read_csv(CSV_PATH + "/" + file)


def plot_graph(df):
    # visualize loss with matplotlib
    # visualize loss with matplotlib
    # visualize loss and acc with matplotlib
    fig, axs = plt.subplots(2)
    axs[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axs[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axs[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    axs[1].plot(df["epoch"], df["val_acc"], label="val_acc")

    # show graph
    plt.legend()
    plt.show()


def main():
    file = get_csv_file()
    df = read_csv_file(file)
    plot_graph(df)


if __name__ == "__main__":
    main()
