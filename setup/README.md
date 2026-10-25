# Cryovision Environment Setup

This project uses **Miniforge** rather than Anaconda to ensure all geospatial packages come exclusively from `conda-forge`, avoiding common DLL and ABI conflicts, and bypassing Anaconda's institutional licensing restrictions.

## Environment Rules
1. **Geospatial Stack from `conda-forge` ONLY**: Packages like GDAL, Rasterio, and GeoPandas must be installed via conda (already configured in the `.yml` files). Never use `pip install gdal` or similar next to a conda environment.
2. **PyTorch from `pip`**: PyTorch and TorchVision are installed via `pip` with the appropriate index URLs to ensure the correct hardware bindings (CUDA for Windows, standard PyPi for MPS).
3. **JupyterLab**: Must be installed and run *inside* the environment, never system-wide.
4. **Windows Path Limits**: Due to the 260-character path limit on Windows, ensure this project is checked out near the drive root (e.g., `C:\dev\cryovision`).
5. **Worker Config on Windows**: DataLoader workers on Windows use `spawn`. Everything inside a worker must be picklable. Ensure `if __name__ == "__main__":` is present in entrypoints.

## Installation Instructions

### Windows (Primary - RTX 4050 / CUDA)
1. Run the Miniforge installer script (requires PowerShell):
   ```powershell
   .\setup\install_miniforge.ps1
   ```
2. Restart your terminal, then initialize conda:
   ```powershell
   conda init powershell
   ```
3. Restart your terminal again and create the environment:
   ```powershell
   conda env create -f setup\environment-cuda.yml
   ```
4. Activate the environment:
   ```powershell
   conda activate cryovision
   ```
5. Verify the installation:
   ```powershell
   python scripts\check_env.py
   ```

### macOS (Secondary - Apple Silicon / MPS)
1. Run the Miniforge installer script:
   ```bash
   bash setup/install_miniforge.sh
   ```
2. Activate and initialize (as directed by the script output):
   ```bash
   source ~/miniforge3/bin/activate
   conda init zsh
   ```
3. Restart your terminal and create the environment:
   ```bash
   conda env create -f setup/environment-mps.yml
   ```
4. Activate the environment:
   ```bash
   conda activate cryovision
   ```
5. Verify the installation:
   ```bash
   python scripts/check_env.py
   ```

### CPU (Fallback)
If you need to install without hardware acceleration:
```bash
conda env create -f setup/environment-cpu.yml
```

> [!CAUTION]
> If `check_env.py` fails on Windows due to PyTorch not finding CUDA (`torch.cuda.is_available() == False`), you likely received a CPU-only wheel. This will run ~30x slower without throwing explicit runtime errors. Re-verify the pip index in `environment-cuda.yml`.
