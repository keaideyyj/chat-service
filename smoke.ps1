param([string]$BaseUrl = "http://127.0.0.1:8000")

function Step {
    param([string]$Name, [scriptblock]$Action)
    try {
        & $Action
        Write-Host "[PASS] $Name" -ForegroundColor Green
    } catch {
        Write-Host "[FAIL] $Name : $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "== $BaseUrl =="

Step "healthz" {
    $r = Invoke-RestMethod "$BaseUrl/healthz" -TimeoutSec 5
    if (-not $r.ok) { throw "ok != true" }
}

Step "config" {
    $r = Invoke-RestMethod "$BaseUrl/config" -TimeoutSec 5
    if (-not $r.api_key_configured) { throw "api_key_configured = false" }
    Write-Host "       model=$($r.model_name)"
}

Step "chat" {
    $body = @{ message = "ping" } | ConvertTo-Json
    $r = Invoke-RestMethod "$BaseUrl/chat" -Method Post -ContentType "application/json" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 60
    if (-not $r.content) { throw "empty content" }
}

Step "chat 拒绝空消息(422)" {
    $status = 0
    try {
        Invoke-WebRequest "$BaseUrl/chat" -Method Post -ContentType "application/json" `
            -Body '{"message":""}' -TimeoutSec 10 | Out-Null
        $status = 200
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
    }
    if ($status -ne 422) { throw "expected 422, got $status" }
}