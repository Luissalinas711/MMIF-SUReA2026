
# Purpose: Startup should rebuild what Google Colab forgets from session to session.
# Finish should save any progress to defined Github Repository
# This file is stored in this repository as a reference for anyone who wants to reproduce this project in their own Colab environment.
# Please adjust necessary values in Colab with your own information


# HOW TO USE IN YOUR OWN COLAB:
# 1. Create a new file in Colab (title mmif-startup or mmif-finish)
# 2. Creat a title block (##Title) cell and insert second cell for code below.
# 3. Tools > Settings > Paste Url into "Custom Notebook Snippet URL" > Save
# 4. Call the startup snippet as Cell 1 in every notebook
# 5. Call finish snippet as the last cell before closing

# Startup meant to run at the beginning of every session
from google.colab import drive, userdata
import os, sys

# Retrieve Github Personal Access Token (PAT) 
# Insert your own PAT in Colab sidebar
PAT = userdata.get('GITHUB_PAT')
print(f'PAT found: {PAT is not None}')

# Mount Google Drive so your image files and models are accessible
drive.mount('/content/drive')

# Settings meant to match your setup
GITHUB_USER = 'Enter Your Username Here'
REPO        = 'MMIF-SUReA2026'
DRIVE_ROOT  = '/content/drive/My Drive/MMIF-SUReA2026'
REPO_PATH   = f'/content/{REPO}'   # where the repo is placed in Colab
DATA        = f'{DRIVE_ROOT}/mmif_data'   # your image files in Google Drive
MODELS      = f'{DRIVE_ROOT}/mmif_models'   # DenseFuse info in Google Drive

# Allows src/ files to be importable by cloning Github repo
os.system(f'git clone https://{PAT}@github.com/'Enter Your Username Here'/MMIF-SUReA2026 {REPO_PATH}')
os.chdir(REPO_PATH)
sys.path.insert(0, f'{REPO_PATH}/src')

# Installs packages (not needed for every session)
os.system('pip install PyWavelets scikit-image -q')

print('Session ready.')

# Finisher meant to run at the end of every session
import subprocess
from datetime import date

# move into the repo folder
os.chdir('/content/MMIF-SUReA2026')
os.system('git config user.name "Enter Your Username Here"')
os.system('git config user.email "Enter Your Username Here@users.noreply.github.com"')

# readies any changed files
os.system('git add .')

# Edit here: short description of changes made this session
subprocess.run(['git', 'commit', '-m', f"{date.today()} — EDIT THIS"])

# Push file(s) to Github
subprocess.run(['git', 'push'])

p = subprocess.run(['git', 'push'], capture_output=True, text=True)
print('Pushed to GitHub.' if p.returncode == 0 else p.stderr)
