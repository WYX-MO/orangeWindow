import torch
print(torch.__version__)  # 查看PyTorch版本
print(torch.cuda.is_available())  # 输出False则表示当前不支持CUDA
print(torch.version.cuda)  # 查看PyTorch绑定的CUDA版本（若为None则未绑定）