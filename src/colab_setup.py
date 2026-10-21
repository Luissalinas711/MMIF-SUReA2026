# Purpose: Startup rebuilds what Google Colab forgets from session to session.
# Finish saves any progress to defined Github Repository
# This file is stored in this repository as a reference for anyone who wants to reproduce this project in their own Colab environment.
# Please adjust necessary values in Colab with your own information
# DenseFuse needs no adjusting, startup fetches the network and weights for you

# HOW TO USE IN YOUR OWN COLAB:
# 1. Create a new file in Colab (title mmif-startup or mmif-finish)
# 2. Create a title block (##Title) cell and insert second cell for code below.
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

# DenseFuse settings, pinned to one commit so every run uses the same model
DENSEFUSE_COMMIT = '4394b63e9295db1c6b7a5c3664551c90f0605f2b'
DENSEFUSE_REPO   = '/content/densefuse-pytorch'   # absolute, the fallback path is relative and breaks after a chdir
DENSEFUSE_CACHE  = f'{DRIVE_ROOT}/densefuse-pytorch'   # the Drive copy, so a new session is a copy and not a clone

# Allows src/ files to be importable by cloning Github repo
os.system(f'git clone https://{PAT}@github.com/{GITHUB_USER}/{REPO} {REPO_PATH}')
os.chdir(REPO_PATH)
sys.path.insert(0, f'{REPO_PATH}/src')

# Installs packages (not needed for every session)
os.system('pip install PyWavelets scikit-image -q')

# Gets the DenseFuse network and pretrained weights, which live outside this repo and are cloned rather than committed since they carry their own license
# Colab empties /content between sessions, so clone once and copy from Drive every time after
if not os.path.isfile(f'{DENSEFUSE_REPO}/net.py'):
    if os.path.isfile(f'{DENSEFUSE_CACHE}/net.py'):
        shutil.copytree(DENSEFUSE_CACHE, DENSEFUSE_REPO, dirs_exist_ok=True)
    else:
        os.system(f'git clone https://github.com/hli1221/densefuse-pytorch {DENSEFUSE_REPO}')
        os.system(f'git -C {DENSEFUSE_REPO} checkout {DENSEFUSE_COMMIT}')
        shutil.copytree(DENSEFUSE_REPO, DENSEFUSE_CACHE,
                        ignore=shutil.ignore_patterns('.git'), dirs_exist_ok=True)

# src/transforms/densefuse.py reads this, so it has to be set before any DenseFuse run
os.environ['DENSEFUSE_REPO'] = DENSEFUSE_REPO

# os.system never raises, so confirm the network and weights are really there instead of failing part way through a run
print('DenseFuse net.py :', os.path.isfile(f'{DENSEFUSE_REPO}/net.py'))
print('DenseFuse weights:', os.path.isfile(f'{DENSEFUSE_REPO}/models/densefuse_gray.model'))

print('Session ready.')


# Finisher meant to run at the end of every session
# Copies every CSV sitting at the top of DRIVE_ROOT into the repo, then commits and pushes.
# Figures are left alone on purpose 
# 07 writes them to DRIVE_ROOT/figures and the repo keeps the script that makes them rather than the images
import subprocess
from datetime import date
from pathlib import Path

# Edit here: short description of changes made this session
MESSAGE = f'{date.today()} EDIT THIS'

results = Path(REPO_PATH) / 'results'
results.mkdir(exist_ok=True)

# The clone puts your PAT in the remote URL, and git quotes that URL back in its errors, so take it out of anything printed
def scrub(text):
    return text.replace(PAT, '<token>') if PAT else text
    
# Anything unexpected stops the cell and says what failed
def git(*args, allow_fail=False):
    done = subprocess.run(['git', '-C', REPO_PATH] + list(args),
                          capture_output=True, text=True)
    if done.returncode != 0 and not allow_fail:
        raise RuntimeError(f'git {" ".join(args)} failed:\n  {scrub(done.stderr.strip())}')
    return done

# Github redirects the clone URL, and git drops credentials on a redirect, so reads still work while a push arrives anonymous. 
# Handing the token to git per command fixes that
def git_with_token(*args):
    helper = f'!f() {{ echo username=x; echo "password={PAT}"; }}; f'

    return git('-c', f'credential.helper={helper}', *args, allow_fail=True)

for source in sorted(Path(DRIVE_ROOT).glob('*.csv')):
    shutil.copy2(source, results / source.name)

git('config', 'user.name', GITHUB_USER)
git('config', 'user.email', f'{GITHUB_USER}@users.noreply.github.com')

# readies any changed files
git('add', '.')
staged = git('diff', '--cached', '--name-only').stdout.split()

if staged:
    print('staged:\n  ' + '\n  '.join(staged))
    git('commit', '-m', MESSAGE)
else:
    print('no CSV changed since the last run')

counted = git('rev-list', '--count', '@{upstream}..HEAD', allow_fail=True)
pending = int(counted.stdout.strip() or 0) if counted.returncode == 0 else 0

if pending == 0:
    print('nothing waiting to push')
else:
    print(f'{pending} commit(s) to push')

    # Precautionary that syncs up Colab and Github before pushing
    rebase = git_with_token('pull', '--rebase')

    if rebase.returncode != 0:
        print(f'rebase failed, fix this before pushing:\n  {scrub(rebase.stderr.strip())}')
    else:
        push = git_with_token('push')
        print('Pushed to GitHub.' if push.returncode == 0 else scrub(push.stderr.strip()))
