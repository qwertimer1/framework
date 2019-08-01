import librosa
import librosa.display as disp
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

class audio_builder():
    def __init__(self, location):
        self.locat = location
        self.count = 0

    def get_audio_and_log_files(self):
        """
        calls get files for wav files and log files and stores these in a list
        """
        self.audiofiles = list(file_utils.get_files(self.locat, extensions = ['.wav'], recurse = True ))
        self.logfiles = list(file_utils.get_files(self.locat, extensions = ['.log'], recurse = True ))
    def _get_audio_snippets(self, audioitem, res, i):
        """
        _get_audio_snippets takes the files found in the compare items and for each item opens the csv file. 
        For each file iterate through rows and feed the start and end times to the _build_audio_item method
        """
        
        for item in res:
            with open(item, 'r') as csvfile:
                
                spamreader = csv.DictReader(csvfile,
                                            fieldnames= ["start time", "end time"], 
                                            delimiter=' ', 
                                            quotechar='|')
                next(spamreader, None)
                for row in spamreader:
                    self._build_audio_item(row, audioitem, i)
    def get_output_filename(self, i):
        """
        builds filename for each file
        """
        
        #print(file.parents[0])
        fn = str(self.audiofilename)
        filename = str(fn.split('.')[0])
        #rmext = str(self.audiofilename.suffix)      
        #a = (fn - rmext)
        self.count = self.count + 1
        res = filename +"_" + str(self.count) + ".wav"
        print(res)
        return res
    def _build_audio_item(self, row, audioitem, i):
        """
        Takes start and end time for each audio item and opens the snippet and saves it to the output wave file.
        """
        #csv file - start time
        start = row['start time']
        #csv file - end time
        end = row['end time']
        start_mod = float(start) - 0.5
        elapsed = float(end) - float(start)
        duration = elapsed + 0.5      
        data, sr = librosa.load(self.audiofilename, offset = start_mod, duration= duration)
        self.get_output_filename(i)
        librosa.output.write_wav(self.get_output_filename(start), data, sr)
    def compare_items(self):
        """
        
        """
        #Compare audiofiles with logfiles to find which logfiles are linked with audiofiles
        i = 0
        for audioitem in self.audiofiles:
           
            self.audiofilename = audioitem
            audioitem = str(audioitem.stem)
            #
            res = [x for x in self.logfiles if re.search(audioitem, str(x))] 
            
            #Calls get audio snippets
            self._get_audio_snippets(audioitem, res, i)

def main():
    location = filedialog.askdirectory()
    ab = audio_builder(location)
    ab.get_audio_and_log_files()
    ab.compare_items()



if __name__ == "__main__":
    main()
