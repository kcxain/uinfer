import os
from loguru import logger

def get_gpu_groups(model_size: int):
    logger.info(f"Using model size: {model_size}")
    gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    gpu_groups = []
    if model_size == 4:
        for gi in gpu_list:
            gpu_groups.append([gi])
    elif model_size == 7 or model_size == 8:
        for i in range(0, len(gpu_list), 2):
            if i + 1 < len(gpu_list):
                gpu_groups.append([gpu_list[i], gpu_list[i + 1]])
            else:
                gpu_groups.append([gpu_list[i]])
    else:
        raise ValueError(
            "Unsupported model size."
        )
    return gpu_groups
