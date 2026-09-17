$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

$ComposeArgs = @("--env-file", ".env.docker.example", "-f", "compose.yaml")
$Fixture = Join-Path ([IO.Path]::GetTempPath()) ("isyara-sign-smoke-{0}.mp4" -f [guid]::NewGuid())
$Body = Join-Path ([IO.Path]::GetTempPath()) ("isyara-sign-response-{0}.json" -f [guid]::NewGuid())

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & docker compose @ComposeArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-Host "Building and starting mock container..."
    Invoke-Compose up --build --detach sign

    $live = $null
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        try {
            $live = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health/live" -ErrorAction Stop
            break
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if ($null -eq $live) { throw "Container did not become reachable within 90 seconds" }
    if ($live.status -ne "alive") { throw "Unexpected liveness response" }

    $ready = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health/ready"
    if ($ready.status -ne "ready") { throw "Mock readiness is not ready" }

    $containerId = (& docker compose @ComposeArgs ps -q sign).Trim()
    $health = $null
    $healthDeadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $healthDeadline) {
        $health = (& docker inspect --format "{{.State.Health.Status}}" $containerId).Trim()
        if ($health -eq "healthy") { break }
        Start-Sleep -Seconds 2
    }
    if ($health -ne "healthy") { throw "Container healthcheck is $health" }

    $published = (& docker compose @ComposeArgs port sign 8765).Trim()
    if ($published -ne "127.0.0.1:8765") {
        throw "Host port is not loopback-only: $published"
    }

    $uid = (& docker compose @ComposeArgs exec --no-TTY sign id -u).Trim()
    if ($uid -eq "0") { throw "Container is running as root" }

    Write-Host "Generating a valid MP4 inside the container..."
    & docker compose @ComposeArgs exec --no-TTY sign ffmpeg -hide_banner -loglevel error -y `
        -f lavfi -i "color=c=black:s=224x224:r=8" -t 1 -pix_fmt yuv420p /tmp/sign-smoke.mp4
    if ($LASTEXITCODE -ne 0) { throw "Could not generate smoke fixture inside container" }
    & docker compose @ComposeArgs cp "sign:/tmp/sign-smoke.mp4" $Fixture
    if ($LASTEXITCODE -ne 0) { throw "Could not copy smoke fixture from container" }

    $httpCode = & curl.exe --silent --show-error --output $Body --write-out "%{http_code}" `
        -X POST "http://127.0.0.1:8765/v1/sign/predict" `
        -H "X-Request-ID: 00000000-0000-0000-0000-000000000001" `
        -F "video=@$Fixture;type=video/mp4" `
        -F "topK=3" `
        -F "vocabularyVersion=mvp-en-v1"
    if ($LASTEXITCODE -ne 0 -or $httpCode -ne "200") {
        throw "Prediction smoke test returned HTTP $httpCode"
    }
    $prediction = Get-Content -Raw -LiteralPath $Body | ConvertFrom-Json
    if ($prediction.modelVersion -ne "mock-sign-v0") {
        throw "Mock smoke test returned modelVersion=$($prediction.modelVersion)"
    }

    $allowedHeaders = & curl.exe --silent --show-error --dump-header - --output NUL `
        -X OPTIONS "http://127.0.0.1:8765/v1/sign/predict" `
        -H "Origin: http://localhost:5173" `
        -H "Access-Control-Request-Method: POST" `
        -H "Access-Control-Request-Headers: content-type,x-request-id"
    if (-not ($allowedHeaders -match "(?im)^access-control-allow-origin:\s*http://localhost:5173")) {
        throw "Allowed-origin CORS preflight did not receive an allow-origin header"
    }

    $pnaHeaders = & curl.exe --silent --show-error --dump-header - --output NUL `
        -X OPTIONS "http://127.0.0.1:8765/v1/sign/predict" `
        -H "Origin: http://localhost:5173" `
        -H "Access-Control-Request-Method: POST" `
        -H "Access-Control-Request-Headers: content-type,x-request-id" `
        -H "Access-Control-Request-Private-Network: true"
    if ($pnaHeaders -match "(?im)^access-control-allow-private-network:") {
        throw "PNA header was returned while SIGN_ALLOW_PRIVATE_NETWORK=false"
    }

    $deniedHeaders = & curl.exe --silent --show-error --dump-header - --output NUL `
        -X OPTIONS "http://127.0.0.1:8765/v1/sign/predict" `
        -H "Origin: http://evil.example" `
        -H "Access-Control-Request-Method: POST" `
        -H "Access-Control-Request-Headers: content-type,x-request-id"
    if ($deniedHeaders -match "(?im)^access-control-allow-origin:") {
        throw "Disallowed-origin CORS preflight received an allow-origin header"
    }

    & docker run --rm --entrypoint sh isyara-sign-service:local -c 'test ! -e /app/.env && test ! -d /app/.venv && test ! -d /app/models'
    if ($LASTEXITCODE -ne 0) { throw "Image-content audit failed: secret, virtualenv, or models directory found" }
    $imageFind = "find /app -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.safetensors' \) -print -quit"
    $unexpectedMedia = & docker run --rm --entrypoint sh isyara-sign-service:local -c $imageFind
    if ($LASTEXITCODE -ne 0 -or $unexpectedMedia) { throw "Image-content audit failed: media or checkpoint found" }

    Write-Host "Docker mock smoke passed: liveness, readiness, prediction, CORS, loopback port, non-root, and image audit."
}
finally {
    try { Invoke-Compose down --remove-orphans } catch { Write-Warning $_ }
    Remove-Item -Force -LiteralPath $Fixture, $Body -ErrorAction SilentlyContinue
}
