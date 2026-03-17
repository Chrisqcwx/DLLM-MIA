import os
import sys

# sys.path.append("../src")

from phoenixcat.launch import wait_gpu_run
from phoenixcat.logger import init_logger

init_logger("wait_gpu_run.log", console_level="INFO", file_level="WARNING")

CMD = ["bash", "attack/run_epoch4.sh"]
gpu_use_num = 1
gpu_ids = "0-7"
threshold_gb = 5
check_interval = 2
confirm_times = 3

wait_gpu_run(
    CMD,
    gpu_use_num,
    gpu_ids,
    threshold_gb,
    check_interval,
    confirm_times,
)
