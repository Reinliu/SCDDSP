import numpy as np
import os
from datetime import datetime

def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule

def create_date_folder(checkpoints_path,name):
    # if not os.path.exists(checkpoints_path):
    #     os.mkdir(checkpoints_path)
    date = datetime.now()
    day = date.strftime('%d-%m-%Y_')
    path = f'{checkpoints_path}/{name}_{day}{str(date.hour)}'
    if not os.path.exists(path):
        os.mkdir(path)
    return path