param(
    [string]$venvPath = ".venv",
    [switch]$install
)

Write-Host "Creating virtual environment at $venvPath..."
python -m venv $venvPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment. Ensure Python is installed and on PATH."
    exit 1
}

$absActivate = Join-Path (Resolve-Path $venvPath) "Scripts\Activate.ps1"
Write-Host "Virtual environment created at: $(Resolve-Path $venvPath)"
Write-Host "To activate in this PowerShell session, run (example):"
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process;`n  & '$absActivate'"

if ($install) {
    Write-Host "Upgrading pip and installing dependencies from requirements.txt..."
    & "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
    & "$venvPath\Scripts\python.exe" -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency installation failed. See output above."
        exit 1
    }
    Write-Host "Dependencies installed. Activate the venv with the command shown above."
}
