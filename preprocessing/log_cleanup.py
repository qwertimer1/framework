\
import mimetypes
import fnmatch
import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from tkinter import filedialog
import re
import csv
from collections import OrderedDict

class file_utils():
    
    @staticmethod
    def listify(o):
        if o is None: return []
        if isinstance(o, list): return o
        if isinstance(o, str): return [o]

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
            

class logfile_reformatter:


    def __init__(self,
                 name = "dummy.txt"):

        self.file_in = ""
        self.file_out = ""
        self.filepath = ""
        self.name = name
        
    def reformatter(self, file):
        """
        reformatter: Method takes in the csv file and removes any excess axis. 
        The file is then rearranged with new headers and resaved
        """
        
        data = pd.read_csv(file, delim_whitespace = True, header = None, error_bad_lines=False)
        data = data.dropna(axis = 1, how = 'any')
        df = data.iloc[:,[0, 1]]
        try:
            a = df[1].str.contains("start")
            if a.empty == False:
                a = df.drop(df.index[[0]])
                df = a
        except:
            pass
        #vals = list(df.columns.values)
        df.columns = ["start time", "end time"]
        
            
        df.to_csv(self.file_out, sep = ' ',index = False)
        return df
    
    def output_file_creator(self):
        """
        Saves the output files as log files.
        
        """
        
        self.file_out = str(self.filepath.joinpath(self.name)) + '.log'
        

    def clean_log_files(self):
        """
        clean_log_files: This is the main function to read in all the log or box files
        and reorganise them in the correct way. 
        The user is asked to provide the directory and the folders were searched for log or box files
        these are modified using reformatter.
        """
        locat = filedialog.askdirectory()
        
        fn = file_utils.get_files(locat, extensions = ['.log', '.box'], recurse = True )
        for file in fn:
            self.name = file.stem
            self.filepath = file.parents[0]                
            self.output_file_creator()            
            self.reformatter(file)

def main():

    lfr = logfile_reformatter()
    lfr.clean_log_files()


if __name__ == "__main__":
    main()