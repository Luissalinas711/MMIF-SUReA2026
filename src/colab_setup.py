
# Purpose: Startup should rebuild what Google Colab forgets from session to session.
# Finish should save any progress to defined Github Repo because Colab does not save automatically.
# This file is stored in this repository as a reference for anyone who wants to reproduce this project in their own Colab environment.
# It is not imported directly from Colab, as errors can occur. 

# HOW TO USE IN YOUR OWN COLAB:
# 1. Create a new file in Colab (title mmif-startup or mmif-finish)
# 2. Creat a title block (##Title) and insert cell for code below.
# 3. Tools > Settings > Paste Url into "Custom Notebook Snippet URL" > Save
# 4. Call the startup snippet as Cell 1 in every notebook
# 5. Call finish snippet as the last cell before closing

# Startup meant to run at the beginning of every session
from google.colab import drive, userdata
import os, sys

PAT = userdata.get('GITHUB_PAT')
print(f'PAT found: {PAT is not None}')

drive.mount('/content/drive')

# Settings meant to match your setup
GITHUB_USER = 'Luissalinas711'
REPO        = 'MMIF-SUReA2026'
DRIVE_ROOT  = '/content/drive/My Drive/MMIF-SUReA2026'
REPO_PATH   = f'/content/{REPO}'
DATA        = f'{DRIVE_ROOT}/mmif_data'
MODELS      = f'{DRIVE_ROOT}/mmif_models'

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
print('Session ready.')
print(f'  DATA   : {DATA}')
print(f'  MODELS : {MODELS}')

# Finisher meant to run at the end of every session
import subprocess
from datetime import date

os.chdir(REPO_PATH)
os.system(f'git config user.name "Luissalinas711"')
os.system('git add .')

commit_msg = f"{date.today()} — EDIT THIS MESSAGE"
r = subprocess.run(['git', 'commit', '-m', commit_msg],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)

p = subprocess.run(['git', 'push'], capture_output=True, text=True)
print('Pushed.' if p.returncode == 0 else p.stderr)
