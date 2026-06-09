
# PURPOSE: This file is stored in this repository as a reference for
# anyone who wants to reproduce this project in their own Colab environment.
# It is not imported directly from Colab, as errors can occur 
# when pulling from 
# HOW TO USE IN YOUR OWN COLAB:
# 1. Copy the startup block into a Colab Snippet (Tools > Snippets)
# 2. Copy the finish() function into a second Colab Snippet
# 3. Call the startup snippet as Cell 1 in every notebook
# 4. Call finish("insert your message here") as the last cell before closing

from google.colab import drive, userdata
import os, sys, subprocess
from datetime import date

# Settings meant to match your setup
GITHUB_USER = 'Luissalinas711'
REPO = 'MMIF-SUReA2026'
DRIVE_ROOT = '/content/drive/My Drive/MMIF-SUReA2026'
REPO_PATH = f'/content/{REPO}'

# Startup meant to run at the beginning of every session
def startup():
 PAT = userdata.get('GITHUB_PAT')
 print(f'PAT found: {PAT is not None}')
 drive.mount('/content/drive')
 if os.path.exists(REPO_PATH):
 os.chdir(REPO_PATH)
 os.system('git pull')
 else:
 result = os.popen(
 f'git clone https://{PAT}@github.com/{GITHUB_USER}/{REPO} '
 f'{REPO_PATH} 2>&1'
 ).read()
 print(result)
 os.chdir(REPO_PATH)
 sys.path.insert(0, REPO_PATH)
 sys.path.insert(0, f'{REPO_PATH}/src')
 os.system('pip install PyWavelets scikit-image -q')
 import builtins
 builtins.DATA = f'{DRIVE_ROOT}/mmif_data'
 builtins.MODELS = f'{DRIVE_ROOT}/mmif_models'
 print('Session ready.')
 print(f' REPO : {REPO_PATH}')
 print(f' DATA : {builtins.DATA}')

# Finisher meant to run at the end of every session
 finish(message=None):
 os.chdir(REPO_PATH)
 os.system(f'git config user.name "{GITHUB_USER}"')
 os.system('git add .')
 msg = message or f"{date.today()} — session commit"
 r = subprocess.run(['git', 'commit', '-m', msg], capture_output=True, text=True)
 print(r.stdout or r.stderr)
 p = subprocess.run(['git', 'push'], capture_output=True, text=True)
 print('Pushed.' if p.returncode == 0 else p.stderr)
