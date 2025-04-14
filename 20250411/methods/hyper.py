import torch

main_folder_name = '20250411'
GPU_COUNT = torch.cuda.device_count()

# 设置训练参数
learning_rate = 1e-3
num_epochs = 200
BATCH_SIZE= 2
BATCH_SIZE_VAL = 1
