# Clean single-puzzle APS measurement. Truncates the puzzle log, launches ARC,
# samples [APS] lines for N seconds, kills it, reports the 1st-percentile APS.
# Usage: measure_aps.ps1 [seconds] [puzzleId]
param([int]$Seconds = 10, [string]$Puz = "06df4c85")
$ErrorActionPreference = "SilentlyContinue"
$log = "logs\$Puz.txt"
taskkill /F /IM ARC.exe /T 2>$null | Out-Null
Start-Sleep -Milliseconds 400
Set-Content -Path $log -Value "" -Encoding utf8
Start-Process -FilePath ".\out\x64-Release\ARC.exe" | Out-Null
Start-Sleep -Seconds $Seconds
$lines = Select-String -Path $log -Pattern '\[APS\]\s+([0-9.]+)'
taskkill /F /IM ARC.exe /T 2>$null | Out-Null
if (-not $lines) { "NO APS LINES"; exit 1 }
# Preserve emission order to drop the warmup ramp, then sort for percentiles.
$ordered = @($lines | ForEach-Object { [double]$_.Matches[0].Groups[1].Value })
$skip = [int][math]::Floor(0.15 * $ordered.Count)
$vals = @($ordered[$skip..($ordered.Count - 1)] | Sort-Object)
$n = $vals.Count
$p1 = $vals[[int][math]::Floor(0.01 * ($n - 1))]
$med = $vals[[int][math]::Floor(0.5 * ($n - 1))]
"samples=$n  p1={0:N0}  median={1:N0}  min={2:N0}  max={3:N0}" -f $p1, $med, $vals[0], $vals[-1]
