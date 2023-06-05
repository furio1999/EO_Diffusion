import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def make_label(shape, mnw, mnh, mxw, mxh):

    label = np.zeros(shape)
    w, h = shape 

    mnw = int(w*mnw/100)
    mxw = int(w*mxw/100)
    mnh = int(h*mnh/100)
    mxh = int(h*mxh/100)

    ws = np.random.randint(mnw, mxw, 1)[0]
    hs = np.random.randint(mnh, mxh, 1)[0]

    x = np.random.randint(ws, w-ws, 1)[0]
    y = np.random.randint(hs, h-hs, 1)[0]

    #print(x,y,ws,hs)

    label[x:x+ws, y:y+hs] = torch.ones((ws,hs))

    return label

#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)