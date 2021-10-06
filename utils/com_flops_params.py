import torch
from thop import profile


def FLOPs_and_Params(model, size, device):
    x = torch.randn(1, 3, size, size).to(device)
    model.trainable = False
    model.eval()

    with torch.no_grad():
        flops, params = profile(model, inputs=(x, ))
    print('- FLOPs : ', flops / 1e6, ' M')
    print('- Params : ', params / 1e6, ' M')

    model.trainable = True
    model.train()

if __name__ == "__main__":
    pass