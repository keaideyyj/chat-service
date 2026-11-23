# run.ps1 —— 开发启动脚本
#
# 作用：先清掉占用端口的残留进程，再启动 uvicorn。
# 为什么需要它：本机不用 --reload（见 DEC-012），改代码后必须手动重启；
#   而反复 Ctrl+C / 端口被僵尸进程占住，是这台机器上最高频的三个坑之一。
# 没有它也能开发，但你会反复手敲两条命令并偶尔撞上 [Errno 10048]。
#
# 用法：
#   .\run.ps1              # 默认 8000
#   .\run.ps1 -Port 8001   # 换端口
# 如果被执行策略拦住：
#   powershell -ExecutionPolicy Bypass -File .\run.ps1

param([int]$Port = 8000)

Set-Location $PSScriptRoot

$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        $proc = Get-Process -Id $_ -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[清理] 端口 $Port 被 PID $($proc.Id) 占用（启动于 $($proc.StartTime)），已结束" -ForegroundColor Yellow
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 800
}

$env:PYTHONUNBUFFERED = "1"

Write-Host "[启动] http://127.0.0.1:$Port  (Ctrl+C 停止)" -ForegroundColor Green
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port $Port
