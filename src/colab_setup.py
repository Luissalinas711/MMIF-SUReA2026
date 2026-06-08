# src/colab_setup.py
# Run with: import colab_setup (start of every session)
# or: colab_setup.finish("insert message here") (end of every session)

from google.colab import drive, userdata
import os, sys, subprocess
from datetime import date

#Edit here to match your setup
GITHUB_USER = 'Luissalinas711'
REPO = 'MMIF-SUReA2026'
DRIVE_ROOT = '/content/drive/My Drive/MMIF-SUReA2026'
REPO_PATH = f'/content/{REPO}'

def _startup():
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
_startup()

#Used at the end of every session
def finish(message=None):
    os.chdir(REPO_PATH)
    os.system(f'git config user.name  "{GITHUB_USER}"')
    os.system('git add .')
    msg = message or f"{date.today()} — session commit"
    r = subprocess.run(['git', 'commit', '-m', msg], capture_output=True, text=True)
    print(r.stdout or r.stderr)
    p = subprocess.run(['git', 'push'], capture_output=True, text=True)
    print('Pushed.' if p.returncode == 0 else p.stderr)
