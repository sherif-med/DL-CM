from lightning.pytorch.callbacks import Callback
import zipfile
import os
from dl_cm import _logger
import tqdm, glob

class ZipFolderCallback(Callback):
    def __init__(self, folder_path, zip_path, glob_pattern="*"):
        """
        Args:
            folder_path (str): Path to the folder that will be zipped.
            zip_path (str): Path where the zip file will be saved, including the name of the zip file.
        """
        self.folder_path = folder_path
        self.zip_path = zip_path
        self.glob_pattern = glob_pattern

    def zip_folder(self):
        """Zips the contents of the folder_path into a zip file saved to zip_path."""
        if os.path.isfile(self.zip_path): os.remove(self.zip_path)
        files = glob.glob(os.path.join(self.folder_path, self.glob_pattern), recursive=True)
        with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Use glob to match files in the folder based on the provided pattern
            for file_path in tqdm.tqdm(files, desc="Zipping prediction"):
                if os.path.isfile(file_path):  # Ensure it's a file
                    # Create a relative path for files to maintain the directory structure
                    relative_path = os.path.relpath(file_path, self.folder_path)
                    zipf.write(file_path, arcname=relative_path)

    
    def on_predict_end(self, trainer, pl_module):
        """Callback function that is called when the training ends."""
        _logger.info(f'Zipping folder {self.folder_path} to {self.zip_path}')
        self.zip_folder()
        _logger.info(f'Folder {self.folder_path} has been successfully zipped to {self.zip_path}')
