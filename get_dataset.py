import kagglehub
import os

cwd = os.getcwd()
os.environ['KAGGLEHUB_CACHE'] = os.getcwd()
path = kagglehub.dataset_download("arnaud58/landscape-pictures")
print(path)