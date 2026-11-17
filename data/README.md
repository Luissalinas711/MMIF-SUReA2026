## Image Files Location
Image files are not stored in this repository.  
Image files are stored in Google Drive due to their size.  
All images are saved as PNG regardless of the original format  

## How to Download
Follow steps in notebooks/01_download_images.py Note that src/colab_setup.py stores a startup and finish cell that should be loaded into most sessions!  
Startup cell initializes everything and finish cell pushes results to repository (read src/colab_setup.py for further information).  
Run to download automatically to appropriate Google Drive location (remember to update startup cell with your information).  
Always remember to give new Colab files access to PAT (this is done manually).  
Manual downloading is an option, but ensure that image sizes and file types are consistent.  

## Modality Combinations (3 Total)
MRI+CT — anatomical + anatomical   
MRI+SPECT — anatomical + functional  
MRI+PET — anatomical + functional  

## Google Drive Location
My Drive/MMIF-SUReA2026/mmif_data/  
mri_ct/ filenames: case16_mri_008.png, case16_ct_008.png  
mri_spect/ filenames: case10_mri_015.png, case10_tc_015.png  
mri_pet/ filenames: caseNN1_mri_008.png, caseNN1_dg_008.png  
The pattern is case, modality, slice number. tc is SPECT and dg is PET.  
 
