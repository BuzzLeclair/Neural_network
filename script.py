import os

def trim_foldernames(directory):
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        if os.path.isdir(folder_path) and '_' in foldername:
            new_foldername = '_'.join(foldername.split('_')[:-1])
            new_folder_path = os.path.join(directory, new_foldername)
            os.rename(folder_path, new_folder_path)
            print(f'Renamed: {folder_path} -> {new_folder_path}')

# Update this path to your local dataset directory
brain_mri_path = './kaggle_3m'

# Call the function to trim folder names
trim_foldernames(brain_mri_path)