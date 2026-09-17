$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

if (-not $env:SIGNBART_REPO_HOST_PATH) {
    $env:SIGNBART_REPO_HOST_PATH = (Join-Path $Project "SignBart")
}

docker compose --env-file .env.docker.example --profile real run --rm --no-deps sign-real `
    python -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device_count': torch.cuda.device_count()}); raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "NVIDIA GPU is not visible inside the container"
}
