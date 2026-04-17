#!/bin/bash

# Set up micromamba if it's not already in the environment
source ./basicMamba.sh  # or source micromamba's init if needed

# Activate your env
micromamba activate zmmg_corrections

python3 -u 10.py "$@"
