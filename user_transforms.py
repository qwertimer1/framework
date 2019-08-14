import torch
import math
from typing import Optional
import matplotlib.pyplot as plt
import PIL


class Transform():
    _order = 0
    def setup(self, dsrc): return    # 1-time setup
    def __call__(self,o):  return o  # transform
    def decode(self,o):    return o  # reverse transform for display


        
class ToCuda(Transform):
    """For Audio data
    Arguments:
        ad: Audio Data
    
    Returns:
        sig = cuda(sig)
        sr = sampling rate
    """
    _order=30
    
    def __call__(self, ad):
        sig = ad
        print("ToCuda")
        return sig.cuda()   
    #def __call__(self, ad):
    #    sig,sr = ad
    #    print("ToCuda")
    #    return (sig.cuda(), sr)
class to_tensor(Transform):
    _order = 1
    def __call__(self, ad, tensor):
        return torch.from_numpy(ad).type_as(tensor)
class to_byte_tensor(Transform):
    _order=10
    def __call__(self, ad):  
       
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(ad.tobytes()))
        w,h = ad.size
        #print('to_byte_tensor')
        #print(type(res))
        return res.view(h,w,-1).permute(2,0,1)
 
class to_byte_tensor_audio(Transform):
    _order=10
    def __call__(self, ad):  
        print(ad)
        audio = ad
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(audio.tobytes()))
        
        #print('to_byte_tensor')
        #print(type(res))
        return res  

class to_float_tensor(Transform):
    _order=20    
    def __call__(self, ad):
        return ad.float().div_(255.)
    


#Image Transforms

class MakeRGB(Transform):
    def __call__(self, item): return item.convert('RGB')
    
class ResizeFixed(Transform):
    _order=10
    def __init__(self,size):
        if isinstance(size,int): size=(size,size)
        self.size = size
        
    def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR)




