import os


def get_gpu_groups(model_size: int):
    if model_size == 7 or model_size == 8:
        # get gpu list
        gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        gpu_groups = []
        for i in range(0, len(gpu_list), 2):
            if i + 1 < len(gpu_list):
                gpu_groups.append([gpu_list[i], gpu_list[i + 1]])
            else:
                gpu_groups.append([gpu_list[i]])
        return gpu_groups
    else:
        raise ValueError(
            "Unsupported model size. Please provide a model size of 7 or 8."
        )
