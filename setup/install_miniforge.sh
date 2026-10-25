#!/bin/bash
# macOS Miniforge installation script

URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
INSTALLER="/tmp/miniforge.sh"

echo "Downloading Miniforge3 for Apple Silicon..."
curl -L $URL -o $INSTALLER
chmod +x $INSTALLER

echo "Installing Miniforge3..."
$INSTALLER -b -p $HOME/miniforge3

echo "Miniforge3 installed successfully."
echo "Run 'source $HOME/miniforge3/bin/activate' and then 'conda init zsh'."
