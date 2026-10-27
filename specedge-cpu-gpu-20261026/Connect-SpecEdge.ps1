param(
    [ValidateSet('cpunode1','cpunode2','cpunode3')]
    [string[]]$CpuNodes = @('cpunode1'),
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$sshExe = (Get-Command ssh.exe -ErrorAction Stop).Source
$baseOptions = @('-N','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3')
$forwardProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
$errorDrains = @{}
$linkLogRoot = Join-Path $PSScriptRoot ('verification\links-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $linkLogRoot -ErrorAction Stop | Out-Null
function Start-Forward([string[]]$ExtraArguments) {
    $linkLog = Join-Path $linkLogRoot ($ExtraArguments[-1] + '.log')
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $sshExe
    $startInfo.Arguments = ($baseOptions + @('-v') + $ExtraArguments) -join ' '
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $process.Start() | Out-Null
    $forwardProcesses.Add($process)
    $pendingLine = $process.StandardError.ReadLineAsync()
    $deadline = (Get-Date).AddSeconds(35)
    while ($true) {
        if ($process.HasExited) { throw "SSH forwarding failed with exit code $($process.ExitCode). Inspect $linkLog" }
        if ($pendingLine.IsCompleted) {
            $details = $pendingLine.GetAwaiter().GetResult()
            if ($null -eq $details) { throw "SSH forwarding closed before becoming ready. Inspect $linkLog" }
            [System.IO.File]::AppendAllText($linkLog, $details + [Environment]::NewLine)
            if ($details -match 'Local forwarding listening on 127\.0\.0\.1 port 19736|remote forward success for:') {
                $errorDrains[$process.Id] = @{ Task = $process.StandardError.ReadToEndAsync(); Path = $linkLog }
                break
            }
            $pendingLine = $process.StandardError.ReadLineAsync()
            continue
        }
        if ((Get-Date) -gt $deadline) { throw "SSH forwarding did not become ready. Inspect $linkLog" }
        Start-Sleep -Milliseconds 250
    }
}
try {
    $portTest = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,19736)
    try { $portTest.Start() } finally { $portTest.Stop() }
    Start-Forward @('-L','127.0.0.1:19736:127.0.0.1:19736','gpunode2')
    foreach ($cpuNode in ($CpuNodes | Select-Object -Unique)) {
        Start-Forward @('-R','127.0.0.1:29736:127.0.0.1:19736',$cpuNode)
    }
    Write-Host ('SpecEdge SSH links ready for: ' + ($CpuNodes -join ', '))
    Write-Host 'Keep this PowerShell window open. Stop with Ctrl+C after the experiment.'
    if (-not $CheckOnly) {
        while ($true) {
            foreach ($process in $forwardProcesses) {
                if ($process.HasExited) { throw "SSH forwarding exited: $($process.ExitCode)" }
            }
            Start-Sleep -Seconds 1
        }
    }
} finally {
    foreach ($process in $forwardProcesses) {
        if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit(5000) | Out-Null }
        if ($errorDrains.ContainsKey($process.Id)) {
            $drain = $errorDrains[$process.Id]
            if ($drain.Task.IsCompletedSuccessfully) { [System.IO.File]::AppendAllText($drain.Path, $drain.Task.GetAwaiter().GetResult()) }
        }
        $process.Dispose()
    }
}
