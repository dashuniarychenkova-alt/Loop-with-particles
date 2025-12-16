Param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\.."),
    [string]$VenvPath,
    [string]$PyInstallerExe
)

if (-not $PyInstallerExe) {
    $candidates = @()
    if ($VenvPath) {
        $candidates += (Join-Path $VenvPath "Scripts\pyinstaller.exe")
    }
    $candidates += (Join-Path $ProjectRoot ".venv-build\Scripts\pyinstaller.exe")
    $candidates += (Join-Path $ProjectRoot ".venv\Scripts\pyinstaller.exe")

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $PyInstallerExe = $candidate
            break
        }
    }
}

if (-not (Test-Path $PyInstallerExe)) {
    Write-Error "PyInstaller executable not found. Bootstrap a virtual environment first:"
    Write-Host "  py -m venv .venv-build"
    Write-Host "  .\.venv-build\Scripts\pip install -r requirements.txt"
    Write-Host "  .\scripts\build-windows.ps1"
    exit 1
}

Push-Location $ProjectRoot
& $PyInstallerExe --noconfirm ".\SpaceParticles.spec"
Pop-Location
