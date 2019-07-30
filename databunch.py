from pathlib import Path
import PIL,os,mimetypes
import librosa
from functools import partial
import random
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler,  DataLoader
from collections import OrderedDict
import re
import transforms

from typing import *
import utils
from utils import util_to_tensor, file_utils, ItemList, CategoryProcessor, LabeledData, parent_labeler





class ImageList(ItemList):
    image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None: extensions = image_extensions
        return cls(file_utils.get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
    
    def get(self, fn): return PIL.Image.open(fn)
    
class AudioList(ItemList):
    audio_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('audio/'))
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None: extensions = audio_extensions
        return cls(file_utils.get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
    
    def get(self, fn): 
        sig,sr = librosa.load(fn)
        sig = util_to_tensor(sig, torch.FloatTensor)
        return (sig, sr)

def random_splitter(fn, p_valid, **kwargs):
    return random.random() < p_valid

     
def split_by_func(items, f):
    mask = [f(o) for o in items]
    # `None` values will be filtered out
    f = [o for o,m in zip(items,mask) if m==False]
    t = [o for o,m in zip(items,mask) if m==True ]
    return f,t

class SplitData():
    def __init__(self, train, valid): self.train,self.valid = train,valid
        
    def __getattr__(self,k): return getattr(self.train,k)
    #This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self,data): self.__dict__.update(data) 
    
    @classmethod
    def split_by_func(cls, il, f):
        lists = map(il.new, split_by_func(il.items, f))
        print(f"Splitting - {lists}")
        return cls(*lists)

    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'
     
    
    def to_databunch(self, sd, bs, c_in = None, c_out = None, n_jobs= 1, **kwargs):
        dls = get_dls(sd.train, sd.valid, bs, n_jobs)
        print(dls)
        return DataBunch(*dls, c_in = c_in, c_out = c_out)








def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train,valid)




def get_dls(train_ds, valid_ds, bs, n_jobs):
    
    train_dl = DataLoader(train_ds, bs, shuffle = True,
                                 num_workers=n_jobs)
    valid_dl = DataLoader(valid_ds, bs*2,
                                   num_workers=n_jobs)
    #test_dl = DataLoader(dataset, bs*2,
                                  #sampler=samplers[2], num_workers=n_jobs)
    dls = train_dl, valid_dl
    return dls
    
class DataBunch():
    def __init__(self, train_dl, valid_dl, c_in=None, c_out = None):
        self.train_dl =train_dl
        self.valid_dl = valid_dl
        self.c_in =c_in
        self.c_out = c_out 
    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset
    




def get_data(path = "E:\\Masters\\Datasets\\Master Whale Sounds\\Master Whale Sounds\\snapshots 3s editable", training_type = "image"):

        #export
    print("start data loading")
    bs = 4
    image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
    audio_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('audio/'))
    #path = "E:\\Masters\\Datasets\\Master Whale Sounds\\Master Whale Sounds\\snapshots 3s editable"

    #all_files = get_files(path, audio_extensions, recurse= True)
    
        #navy
        # print(f"length of data = {len(il)}")
        #print(il)
 
    print("image files")
    tfms = [transforms.MakeRGB(), transforms.to_byte_tensor(), transforms.to_float_tensor()]
    
    il = ImageList.from_files(path, image_extensions, tfms = tfms)
        #print(al)
    #samplers = train_test_valid_split(il)
    #speccer = Spectrogrammer(to_db=True, n_fft=1024, n_mels=64, top_db=80)
    
    dataset = il #if training_type == "image"
    print("processing...") 
    sd = SplitData.split_by_func(dataset, partial(random_splitter, p_valid = 0.3))
    #print(sd.train)
    ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
   
    #databunchify(dataset, samplers, bs)

    ll = ll.to_databunch(ll, bs, c_in=3, c_out=12, num_workers=4)
    return(ll)



def main():
    ll = get_data()
#### CURRENT CODE CREATES A NICE LITTLE AUDIOLIST OR IMAGE LIST
if __name__ == "__main__":
    main()