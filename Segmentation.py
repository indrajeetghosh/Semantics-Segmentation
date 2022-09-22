import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch
from Decode_Segmentation import decode_segmap

def segment(net, path, show_orig=True, dev='cuda'):
    img = Image.open('Images/bird.png')
    if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
    trf = T.Compose([T.Resize(640), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb); plt.axis('off'); plt.show()