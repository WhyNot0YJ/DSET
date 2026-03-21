import torch
import PIL.Image as Image
import torchvision.transforms.v2 as T
from experiments.cas_detr.src.data.transforms._transforms import PadToSize

img = torch.rand(3, 1080, 1920)
trans = T.Compose([
    T.Resize(size=640, max_size=640, antialias=True),
    PadToSize(size=(640, 640), fill=114)
])
out = trans(img)
print(out.shape)
