import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from swin_transformer import SwinTransformer

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

model = SwinTransformer()
target_layers = [model.layers[-1].blocks[-1].norm1]
input_tensor = torch.ones((2,3,224,224))

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
targets = [ClassifierOutputTarget(1)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

print(grayscale_cam.shape)
