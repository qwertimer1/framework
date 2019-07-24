import json
from torch.utils.data import DataLoader
from pathlib import Path
import datasets
from utils import utils

class datasets_test_harness():
    def __init__(self, dataset = "",**params):
    
        self.data_generator = DataLoader(dataset, **params)
        self.dl_iter = iter(self.data_generator)

    def iter_data(self):
        for i in range(5):
            data = next(self.dl_iter)
            print(f"data = {data}")
        
    
    def test_loader(self):
        for batch in self.data_generator:
            x,y = batch
            print(f"x = {x}")
            print(f"y = {y}")

    def harness(self):
        print("Testing Dataset")
        print("test 1")
        self.iter_data()

        print("test 2")
        self.test_loader()



def main():
    config_file = "training_config.json"
    with open(config_file, "r") as f:
        x = json.load(f)


    #hyperparams = x["HYPERPARAMETERS"]
    #name = x["model"]
    #n_classes = hyperparams['n_classes']
    #n_bands = hyperparams['n_bands']
    #print(n_bands)
    #patch_size = hyperparams['patch_size']
    _, path = utils.experiment_path_build(x)   
    csv_file = str(path) +'/' + 'audio_manifest.csv' 
    dataset = datasets.Whale_Audio_Dataset(path, csv_file)
    test = datasets_test_harness(dataset)
    test.harness()
    
if __name__ == "__main__":
    main()
    