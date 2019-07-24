import torch
from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


from PIL import Image

import librosa
from utils import utils

from pathlib import Path


class Whale_Image_Dataset(Dataset):

    def __init__(self, path, csv_file, transform = None):

        """Whale Image Dataset



        """
    
        self.path = path
        self.transform = transform
        self.csv_file = csv_file
        csv_path = Path(self.path, self.csv_file)
        if not (csv_path.is_file()):
            csvfile = utils.make_manifest(path = self.path,manifest_name = self.csv_file, 
            ext = '.jpg', 
            headerfile=False)
            csv_path = csvfile.name 
        self.data_info = pd.read_csv(csv_path)
        self.filenames = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        le = LabelEncoder()
        self.label_arr_enc = le.fit_transform(self.label_arr)
        torch.from_numpy(self.label_arr_enc).type(torch.LongTensor)

        self.image_names = np.asarray(self.data_info.iloc[:,2])



    
    def __getitem__(self, index):
        data = 0
        im = (self.filenames[index] +'\\'+ self.image_names[index])
        img = Image.open(im)
        img = np.asarray(img)
        data = self.toTensor(img)
        label = self.label_arr_enc[index]

        if self.transform is not None:
            img = self.transform(img)

        return (data, label)

    def __len__(self):
        return(len(self.filenames))

    @staticmethod
    def toTensor(img):
        """convert a numpy array of shape HWC to CHW tensor"""
        img = img.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(img).float()
        return tensor/255.0


class Whale_Audio_Dataset(Dataset):

    
    def __init__(self, path, csv_file):

        """Whale Image Dataset



        """

        self.path = path
        self.csv_file = csv_file
        csv_path = Path(self.path, self.csv_file)
        if not (csv_path.is_file()):
            csvfile = utils.make_manifest(self.path,
            manifest_name = 'manifest.csv', 
            ext = '.wav', 
            headerfile=False)
            csv_path = csvfile.name 
        self.data_info = pd.read_csv(csv_path)
        self.filenames = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        le = LabelEncoder()
        self.label_arr_enc = le.fit_transform(self.label_arr)
        torch.from_numpy(self.label_arr_enc).type(torch.LongTensor)

        self.audio_names = np.asarray(self.data_info.iloc[:,2])
        
    def __getitem__(self, index):
        data = 0
        fname = self.filenames[index]+'\\'+ self.audio_names[index]
        tmp, sr = librosa.load(fname)

        data = torch.tensor(tmp).float()
        label = self.label_arr_enc[index]

        return (data, label)

    def __len__(self):
        return(len(self.filenames))

    
     
    def get_audio(fname, offset, window_size):
        """Gets windowed audio snippet
        
        Arguments:
            fname: audio file 
            offset: start of audiofile
            window_size: length of audio snippet
        returns:
            data: output audio
            sr: Sampling Rate    
        """
        
        data, sr = librosa.load(fname, sr = None, offset = offset, duration = window_size)
        return(data, sr)
    



    def window_mask(self):
        datavals = []
        results = []
        self.data= torch.tensor(self.data)
        start_val = 0
        occurences = int(len(self.data)/(self.seq_len*self.overlap))
        
        for rows in range(len(self.data)):

            start_val = 0
            for i in range(occurences-1):
                
                    value = self.data[rows,start_val:(start_val+self.seq_len)]
                    
                    start_val += int(self.seq_len*(1 - self.overlap))
                    datavals.append(value.unsqueeze(0))

            var = torch.cat(datavals)        
            results.append(var.unsqueeze(0))
            
            datavals = []
        result = torch.cat(results)
        result = result.permute(2,1,0)
        print(result.shape)
        print(result)
        return result



        
class DatasetLSTM(Dataset):

    """
        Support class for the loading and batching of sequences of samples

        Args:
            dataset (Tensor): Tensor containing all the samples
            sequence_length (int): length of the analyzed sequence by the LSTM
            transforms (object torchvision.transform): Pytorch's transforms used to process the data
    """

    ##  Constructor
    def __init__(self, dataset, sequence_length=1, transforms=None):


        self.dataset = dataset
        self.seq_length = sequence_length
        self.transforms = transforms

    def build_dataset(self, ):
        self.path = path
        self.csv_file = csv_file
        csv_path = Path(self.path, self.csv_file)
        if not (csv_path.is_file()):
            csvfile = utils.make_manifest(self.path,
            manifest_name = 'manifest.csv', 
            ext = '.wav', 
            headerfile=False)
            csv_path = csvfile.name 
        self.data_info = pd.read_csv(csv_path)
        self.filenames = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        le = LabelEncoder()
        self.label_arr_enc = le.fit_transform(self.label_arr)
        torch.from_numpy(self.label_arr_enc).type(torch.LongTensor)

        self.audio_names = np.asarray(self.data_info.iloc[:,2])
        

    ##  Override total dataset's length getter
    def __len__(self):
        return self.dataset.__len__()

    ##  Override single items' getter
    def __getitem__(self, idx):
        if idx + self.seq_length > self.__len__():
            item = []
            item[:self.__len__()-idx] = self.dataset[idx:]
            return item, item
        else:
           
            return self.dataset[idx:idx+self.seq_length], self.dataset[idx:idx+self.seq_length]


###   Helper for transforming the data from a list to Tensor

def listToTensor(list):
    tensor = torch.empty(list.__len__(), list[0].__len__())
    for i in range(list.__len__()):
        tensor[i, :] = torch.FloatTensor(list[i])
    return tensor