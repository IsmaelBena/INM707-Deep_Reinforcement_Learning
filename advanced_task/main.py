import torch
import gymnasium

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Device: {device}')

env = gymnasium.make("CarRacing-v2")

print(f'A: {env}')