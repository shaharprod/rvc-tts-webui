$ErrorActionPreference = "Stop"

try {
    $projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $projectRoot

    $pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"
    $appPath = Join-Path $projectRoot "app.py"
    $port = 7865

    if (-not (Test-Path $pythonExe)) {
        throw "Missing venv python: $pythonExe"
    }
    if (-not (Test-Path $appPath)) {
        throw "Missing app.py: $appPath"
    }

    Write-Host "Project root: $projectRoot"
    Write-Host "Checking port $port..."

    $listenerPids = @()

    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($listeners) {
        $listenerPids += ($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    }

    # Fallback for environments where Get-NetTCPConnection can miss listeners
    $netstatLines = netstat -ano | Select-String ":$port"
    if ($netstatLines) {
        foreach ($line in $netstatLines) {
            $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
            if ($parts.Count -ge 5) {
                $state = $parts[3]
                $owningPid = $parts[4]
                if ($state -eq "LISTENING" -and $owningPid -match "^\d+$") {
                    $listenerPids += [int]$owningPid
                }
            }
        }
    }

    $listenerPids = $listenerPids | Sort-Object -Unique

    if ($listenerPids.Count -gt 0) {
        foreach ($procId in $listenerPids) {
            try {
                Write-Host "Stopping process on port $port (PID: $procId)..."
                Stop-Process -Id $procId -Force -ErrorAction Stop
            }
            catch {
                Write-Warning "Could not stop PID ${procId}: $($_.Exception.Message)"
            }
        }
        Start-Sleep -Milliseconds 700
    }

    Write-Host "Starting RVC TTS WebUI..."
    Write-Host "URL: http://127.0.0.1:$port"
    & $pythonExe $appPath
}
catch {
    Write-Error $_.Exception.Message
}
