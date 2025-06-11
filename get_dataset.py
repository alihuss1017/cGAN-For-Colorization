import kagglehub
import os
import shutil

cwd = os.getcwd()
os.environ['KAGGLEHUB_CACHE'] = os.getcwd()
path = kagglehub.dataset_download("arnaud58/landscape-pictures")

source_folder = path
destination_folder = "data"

os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)

    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)

print(f"Copied all files to {destination_folder}")
