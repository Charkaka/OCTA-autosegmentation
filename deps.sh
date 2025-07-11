#!/bin/sh

# To  run: ./deps.sh
# If you get "permission denied" error, run this command first: chmod +x deps.sh
# then try the command again.

# Install  pip requirements
pip install -r requirements.txt

#  Install diffusers
conda install -y -c conda-forge diffusers

# Install torch and dependencies
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121