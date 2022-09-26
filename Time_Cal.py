import time
from PIL import Image
import torchvision.transforms as T

def infer_time(net, path='Images/bird.png', dev='cuda'):
    img = Image.open('Images/bird.png')
    trf = T.Compose([T.Resize(640), 
                   T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  
    inp = trf(img).unsqueeze(0).to(dev)
  
    st = time.time()
    out1 = net.to(dev)(inp)
    et = time.time()
  
    return et - st
