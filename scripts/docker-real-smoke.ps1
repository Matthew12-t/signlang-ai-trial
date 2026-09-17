$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

$ComposeArgs = @("--env-file", ".env.docker.example", "--profile", "real", "-f", "compose.yaml")
$Fixture = Join-Path ([IO.Path]::GetTempPath()) ("isyara-sign-real-smoke-{0}.mp4" -f [guid]::NewGuid())
$Body = Join-Path ([IO.Path]::GetTempPath()) ("isyara-sign-real-response-{0}.json" -f [guid]::NewGuid())

try {
    $ready = $null
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline) {
        try {
            $ready = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health/ready" -ErrorAction Stop
            if ($ready.status -eq "ready") { break }
        }
        catch {
            # Startup can take several seconds while CUDA and SignBart warm up.
        }
        Start-Sleep -Seconds 2
    }
    if ($null -eq $ready) { throw "Real SignBart readiness endpoint did not become reachable within 120 seconds" }
    if ($ready.status -ne "ready" -or $ready.checks.model -ne "ready" -or $ready.checks.device -ne "cuda:0") {
        throw "Real SignBart readiness is not valid: $($ready | ConvertTo-Json -Compress)"
    }

    & docker compose @ComposeArgs exec --no-TTY sign-real ffmpeg -hide_banner -loglevel error -y `
        -f lavfi -i "color=c=black:s=224x224:r=8" -t 1 -pix_fmt yuv420p /tmp/sign-real-smoke.mp4
    if ($LASTEXITCODE -ne 0) { throw "Could not generate real-model fixture inside container" }

    & docker compose @ComposeArgs cp "sign-real:/tmp/sign-real-smoke.mp4" $Fixture
    if ($LASTEXITCODE -ne 0) { throw "Could not copy real-model fixture from container" }

    $httpCode = & curl.exe --silent --show-error --output $Body --write-out "%{http_code}" `
        -X POST "http://127.0.0.1:8765/v1/sign/predict" `
        -H "X-Request-ID: 00000000-0000-0000-0000-000000000002" `
        -F "video=@$Fixture;type=video/mp4" `
        -F "topK=3" `
        -F "vocabularyVersion=mvp-en-v1"
    if ($LASTEXITCODE -ne 0 -or $httpCode -ne "200") {
        throw "Real prediction smoke test returned HTTP $httpCode"
    }

    $prediction = Get-Content -Raw -LiteralPath $Body | ConvertFrom-Json
    if ($prediction.modelVersion -ne "signbart-wlasl100-hf-8f98ae7") {
        throw "Unexpected real modelVersion=$($prediction.modelVersion)"
    }
    Write-Host "Docker real SignBart smoke passed: CUDA readiness and prediction contract."
    $prediction | ConvertTo-Json -Compress
}
finally {
    Remove-Item -Force -LiteralPath $Fixture, $Body -ErrorAction SilentlyContinue
}
