# Windows Miniforge installation script
$url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
$installer = "$env:TEMP\Miniforge3-Windows-x86_64.exe"

Write-Host "Downloading Miniforge3 for Windows..."
Invoke-WebRequest -Uri $url -OutFile $installer

Write-Host "Installing Miniforge3 (JustMe, non-interactive)..."
Start-Process -FilePath $installer -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /S /D=$env:USERPROFILE\Miniforge3" -Wait -NoNewWindow

Write-Host "Miniforge3 installed successfully."
Write-Host "Please restart your shell and run 'conda init powershell'."
