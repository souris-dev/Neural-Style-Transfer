# Download and prepare the MS-COCO dataset
import os
from sys import platform
import requests

from zipfile import ZipFile

def download_mscoco(download_folder: str) -> str:
    if os.path.isfile(os.path.join(download_folder, 'train_coco.zip')):
        # If training set .zip is already downloaded, do nothing
        return os.path.join(download_folder, 'train_coco.zip')
    elif os.path.isfile(os.path.join(download_folder, 'train2014.zip')):
        return os.path.join(download_folder, 'train2014.zip')

    url = 'http://images.cocodataset.org/zips/train2014.zip'
    # If running on linux, use wget
    if platform.startswith('linux'):
        print('Downloading training data (MS-COCO dataset) using wget...')
        print("Executing: " + "wget " + url + " -O " + os.path.join(download_folder, 'train_coco.zip'))
        if (os.system("wget " + url + " -O " + os.path.join(download_folder, 'train_coco.zip')) != 0):
            print('Training data download failed! Exiting...')
            exit()

    else:
        print('Downloading training data...')
        print('Beginning file download with requests module')
        r = requests.get(url)

        with open(os.path.join(download_folder, 'train_coco.zip'), 'wb') as f:
            f.write(r.content)
    
    return os.path.join(download_folder, 'train_coco.zip')

def extract_mscoco(filename: str):
    fname = filename
    
    with ZipFile(fname, 'r') as zipf:
        zipf.extractall(path=os.path.join(os.getcwd(), 'mscoco/train'))

    return os.path.join(os.getcwd(), 'mscoco')