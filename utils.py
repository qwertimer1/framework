from Callback import Callback
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict, OrderedDict

import os
import sys
from pathlib import Path
import csv
from typing import *
import torch

class ProgressBar(Callback):

    def training_started(self, phases, **kwargs):
        bars = OrderedDict()
        for phase in phases:
            bars[phase.name] = tqdm(total=len(phase.loader), desc=phase.name)
        self.bars = bars

    def batch_ended(self, phase, **kwargs):
        bar = self.bars[phase.name]
        bar.set_postfix_str(f'loss: {phase.last_loss:.4f}')
        bar.update(1)
        bar.refresh()

    def epoch_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = 0
            bar.refresh()

    def training_ended(self, **kwargs):
        for bar in self.bars.values():
            bar.n = bar.total
            bar.refresh()
            bar.close()


class utils():
    def __init__(self):
        pass

    @staticmethod
    def get_experiment_directory():
        """
        Catch all to allow for use on:

        Work Machine - 
        H:/Masters/
        Home Desktop - 
        E:\Masters
        Home Linux -

        Home Laptop -


        Work Linux -
        """
        #System Check

        if os.name == 'nt':
            if os.path.exists('E:/Masters'):
                ROOT = Path('E:/Masters')
            elif os.path.exists('H:/Masters'):
                ROOT = Path('H:/Masters')
            elif os.path.exists('D:/Masters'):
                ROOT = Path('D:\Masters')
            elif os.path.exists("F:/Masters"):
                ROOT = Path('F:\Masters')
            else:
                print('Folder not found')
        else:
            if os.path.exists('\\media\\tim\\'):
                ROOT = '\\media\\tim\\Masters\\'
        #Get Major branch location         
        branch = Path('Datasets\\Master Whale Sounds\\Master Whale Sounds')
        #location = Path(branch)
        #Get leaf locations

        processed = Path('Snapshots 3s editable')
        location = ROOT.joinpath(branch)
        audio_snippets_folders = location.joinpath(processed)
        return(ROOT, audio_snippets_folders) 

    @staticmethod
    def get_classes(path):
        res = []
        dirs = [x for x in path.iterdir() if x.is_dir()]
        for cls in dirs:
            val = os.path.basename(cls)
            res.append(val)
        classes = res
        return dirs, classes
    @staticmethod
    def get_folders(x):
        base = x["LOCATIONS"]
        ROOT = base["ROOT"]
        project = base["project"]
        dataset = base["dataset"]
        return ROOT, project, dataset
    
    @staticmethod
    def experiment_path_build(x):
        ROOT, project, dataset = utils.get_folders(x)
        path = Path(ROOT).joinpath(project).joinpath(dataset)
        return ROOT, path

    @staticmethod    
    def make_manifest(path, manifest_name = 'manifest.csv', ext = '.wav', headerfile = False):
    
        """
        make_manifest: builds a csv file containing the directory, labels and filenames of all the files in the dataset
        path
        manifest_name: csv file name of the manifest
        ext: file extension of items (used to check all files are of the same type
        headerfile: Flag for if there is a header file linked to the main file
        returns:
        csvfile = csvfile name
   
        """
        j = 0
    
        with open(str(path) +'/'+ manifest_name, 'w') as csvfile:
            print("Building Manifest")
            if headerfile == True:
                hdr = '.hdr'
                fieldnames = ['directory', 'label', 'file', 'header file']
            else:
                fieldnames = ['directory', 'label', 'file']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
     
            dirs, classes = utils.get_class_labels(path)
            writer.writeheader()
            for i, dir in enumerate(dirs):
                cls = classes[i]
                if headerfile == True:
            
                    hdr_files = [f for f in os.listdir(str(dir)) if os.path.isfile(os.path.join(str(dir), f)) and f.endswith(hdr)]
                    hdr_files = sorted(hdr_files)
                files = [f for f in os.listdir(str(dir)) if os.path.isfile(os.path.join(str(dir), f)) and f.endswith(ext)]
                folder_size = len(files)
                files = sorted(files)

                for val in range(folder_size):
                
                    if headerfile == True:
                        writer.writerow({'directory' : dir, 'label': cls, 'file' : files[val], 'header file' : hdr_files[val]})
                    else:
                        writer.writerow({'directory' : dir, 'label': cls, 'file' : files[val]})
                    j = j + 1

        print(f'saved {j} files')
        return csvfile
    @staticmethod
    def get_class_labels(path):

        #Populate classes and directory locations
        res = []

        dirs = [x for x in path.iterdir() if x.is_dir()]

        for cls in dirs:
            val = os.path.basename(cls)
            res.append(val)
        classes =res
        return dirs, classes    




class file_utils():

    @staticmethod
    def listify(o):
        if o is None: return []
        if isinstance(o, list): return o
        if isinstance(o, str): return [o]
        if isinstance(o, Iterable): return list(o)
        return [o]
    @staticmethod
    def uniqueify(x, sort=False):
        res = list(OrderedDict.fromkeys(x).keys())
        if sort: res.sort()
        return res

    @staticmethod
    def setify(o): return o if isinstance(o,set) else set(file_utils.listify(o))



    @staticmethod
    def _get_files(p, fs, extensions=None):
        p = Path(p)
        res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
        return res

    @staticmethod
    def get_files(path, extensions=None, recurse=False, include=None):
        path = Path(path)
        extensions = file_utils.setify(extensions)
        extensions = {e.lower() for e in extensions}

        if recurse:

            res = []
            for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
                if include is not None and i==0: d[:] = [o for o in d if o in include]
                else:                            d[:] = [o for o in d if not o.startswith('.')]

                res += file_utils._get_files(p, f, extensions)

            return res
        else:
            f = [o.name for o in os.scandir(path) if o.is_file()]
            return file_utils._get_files(path, f, extensions)
def util_to_tensor(ad, tensor):
       
    return torch.from_numpy(ad)

def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(file_utils.listify(funcs), key=key): x = f(x, **kwargs)
    return x


class ListContainer():
    def __init__(self, items): self.items = file_utils.listify(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res


class ItemList(ListContainer):
    def __init__(self, items, path='.', tfms=None):
        super().__init__(items)
        self.path,self.tfms = Path(path),tfms

    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'
    
    def new(self, items, cls=None):
        if cls is None: cls=self.__class__
        return cls(items, self.path, tfms=self.tfms)
    
    def  get(self, i): return i
    def _get(self, i): return compose(self.get(i), self.tfms)
    
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res,list): return [self._get(o) for o in res]
        return self._get(res)




class Processor(): 
    def process(self, items): return items

class CategoryProcessor(Processor):
    def __init__(self): self.vocab=None
    
    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            self.vocab = file_utils.uniqueify(items)
            self.otoi  = {v:k for k,v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return self.otoi[item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return self.vocab[idx]

#export
def parent_labeler(fn): return fn.parent.name

def _label_by_func(ds, f, cls=ItemList): return cls([f(o) for o in ds.items], path=ds.path)

#This is a slightly different from what was seen during the lesson,
#   we'll discuss the changes in lesson 11
class LabeledData():
    def process(self, il, proc): return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x,self.y = self.process(x, proc_x),self.process(y, proc_y)
        self.proc_x,self.proc_y = proc_x,proc_y
        
    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx): return self.x[idx],self.y[idx]
    def __len__(self): return len(self.x)
    
    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)
    
    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(file_utils.listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)
