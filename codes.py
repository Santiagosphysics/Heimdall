import torch

if torch.cuda.is_available():
    print('CUDA ES DISPONIBLE, UTILIZANDO GPU')
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU en uso {gpu_name}')
else:
    print('No se usa GPU :(')