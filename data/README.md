## Image Files Location
Image files are not stored in this repository.  
Image files are stored in Google Drive due to their size.  
All images are saved as PNG regardless of the original format  

## How to Download
Follow steps in notebooks/01_download_images.py
Note that src/colab_setup.py stores a startup and finish cell that should be loaded into most sessions!  
Startup cell initializes everything and finish cell pushes results to repository (read src/colab_setup.py for further information)  
Run to download automatically to appropriate Google Drive location (remember to update startup cell with your information).  
Always remember to give new Colab files access to PAT (this is done manually).  
Manual downloading is an option, but ensure that image sizes and file types are consistent.

## Modality Combinations (3 Total)
MRI+CT — anatomical + anatomical   
MRI+SPECT — anatomical + functional  
MRI+PET — anatomical + functional  

## Google Drive Location
My Drive/MMIF-SUReA2026/mmif_data/  
 mri_ct/ filenames: label_mri_image#.gif, label_ct_image#.png    
 mri_spect/ filenames: label_mri_image#.gif, label_tc_image#.png    
 mri_pet/ filenames: label_mri_image#.gif, label_dg_image#.png    
 
