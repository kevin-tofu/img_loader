import sys
sys.dont_write_bytecode = True

import numpy as np
import os
import shutil

def make_directory(path):

    print(path)
    if os.path.exists(path) == False:
        os.makedirs(path)

    else:
        #os.rmdir(path)        
        #shutil.rmtree(path)
        pass

def remove_files(directory):

    if os.path.exists(directory) == True:
        shutil.rmtree(directory)

