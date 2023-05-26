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
CSV_PATH = os.path.join(CURRENT_PATH, "csv")


# get csv file and create dataframe
def get_csv_file():
    csv_file = os.listdir(CSV_PATH)
    csv_file = sorted(csv_file)
    return csv_file[-1]


def main():
    file = get_csv_file()

    # from csv and create dataframe
    df = pd.DataFrame(pd.read_csv(os.path.join(CSV_PATH, file)))

    # visualize loss with matplotlib
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")

    # visualize acc with matplotlib
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")

    # show graph
    plt.show()

    return None
