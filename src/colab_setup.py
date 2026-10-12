# Purpose: Startup rebuilds what Google Colab forgets from session to session.
# Finish saves any progress to defined Github Repository
# This file is stored in this repository as a reference for anyone who wants to reproduce this project in their own Colab environment.
# Please adjust necessary values in Colab with your own information
# DenseFuse needs no adjusting, startup fetches the network and weights for you


# HOW TO USE IN YOUR OWN COLAB:
# 1. Create a new file in Colab (title mmif-startup or mmif-finish)
# 2. Creat a title block (##Title) cell and insert second cell for code below.
# 3. Tools > Settings > Paste Url into "Custom Notebook Snippet URL" > Save
# 4. Call the startup snippet as Cell 1 in every notebook
# 5. Call finish snippet as the last cell before closing

# Startup meant to run at the beginning of every session
from google.colab import drive, userdata
import os, shutil, sys

# Retrieve Github Personal Access Token (PAT)
# Insert your own PAT in Colab sidebar
# ALWAYS REMEMBER TO GIVE EACH NEW FILE ACCESS TO YOUR PAT IN COLAB
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

# DenseFuse settings, pinned to one commit so every run uses the same model
DENSEFUSE_COMMIT = '4394b63e9295db1c6b7a5c3664551c90f0605f2b'
DENSEFUSE_REPO   = '/content/densefuse-pytorch'   # absolute, the fallback path is relative and breaks after a chdir
DENSEFUSE_CACHE  = f'{MODELS}/densefuse-pytorch'   # the Drive copy, so a new session is a copy and not a clone

# Allows src/ files to be importable by cloning Github repo
os.system(f'git clone https://{PAT}@github.com/{GITHUB_USER}/{REPO} {REPO_PATH}')
os.chdir(REPO_PATH)
sys.path.insert(0, f'{REPO_PATH}/src')

# Installs packages (not needed for every session)
os.system('pip install PyWavelets scikit-image -q')

# Gets the DenseFuse network and pretrained weights, which live outside this repo and are
# cloned rather than committed since they carry their own license
# Colab empties /content between sessions, so clone once and copy from Drive every time after
if not os.path.isfile(f'{DENSEFUSE_REPO}/net.py'):
    if os.path.isfile(f'{DENSEFUSE_CACHE}/net.py'):
        shutil.copytree(DENSEFUSE_CACHE, DENSEFUSE_REPO, dirs_exist_ok=True)
    else:
        os.system(f'git clone https://github.com/hli1221/densefuse-pytorch {DENSEFUSE_REPO}')
        os.system(f'git -C {DENSEFUSE_REPO} checkout {DENSEFUSE_COMMIT}')
        # the Drive copy skips .git, slow to sync and not needed once the commit is pinned
        shutil.copytree(DENSEFUSE_REPO, DENSEFUSE_CACHE,
                        ignore=shutil.ignore_patterns('.git'), dirs_exist_ok=True)

# src/transforms/densefuse.py reads this, so it has to be set before any DenseFuse run
os.environ['DENSEFUSE_REPO'] = DENSEFUSE_REPO

# os.system never raises, so confirm the weights are really there instead of failing mid run
weights_ready = os.path.isfile(f'{DENSEFUSE_REPO}/models/densefuse_gray.model')
print(f'DenseFuse found: {weights_ready}')

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
subprocess.run(['git', 'commit', '-m', f"{date.today()}: EDIT THIS"])

# Precautionary that syncs up Colab and Github
os.system('git pull --rebase')

p = subprocess.run(['git', 'push'], capture_output=True, text=True)
print('Pushed to GitHub.' if p.returncode == 0 else p.stderr)
