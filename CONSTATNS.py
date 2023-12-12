import os
from tqdm import *
import torch
import numpy as np
import hashlib
import time
import random


print("Seeding everything...")
seed = 6
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # Torch CPU random seed module.
torch.cuda.manual_seed(seed)  # Torch GPU random seed module.
torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = str(seed)

print("Seeding Finished")

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SESSION = hashlib.md5(
    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 60 * 60)).encode(
        "utf-8"
    )
).hexdigest()[:8]
SESSION = "SESSION_" + SESSION

current_abspath = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(current_abspath))
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")
