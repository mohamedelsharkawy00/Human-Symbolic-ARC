param(
    [Parameter(Mandatory=$true)][string]$Vtune,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    # The puzzles to read, every training puzzle where none are named.
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Ids = @()
)
$ErrorActionPreference = 'Stop'
$runDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$scorePath = Join-Path $runDirectory 'score.txt'
# Run from the repo root under the label "profile": its readouts land in
# out/profile, and what it prints is kept here beside the profile.
$runner = Start-Process -FilePath "$PSScriptRoot/../out/x64-RelWithDebInfo/arc_said.exe" -ArgumentList (@('--as','profile') + $Ids) -WorkingDirectory "$PSScriptRoot/.." -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDirectory 'said.txt') -RedirectStandardError $scorePath
try {
    $runner.PriorityClass = 'AboveNormal'
    # Let CUDA initialize before VTune injects its sampler. The score still
    # covers every puzzle; the profile excludes startup and attachment time.
    $startup = [Diagnostics.Stopwatch]::StartNew()
    while (!(Select-String -LiteralPath $scorePath -Pattern 'dataset flags' -Quiet)) {
        if ($runner.HasExited) { throw 'arc_said exited before profiler attachment.' }
        if ($startup.Elapsed.TotalSeconds -gt 30) { throw 'arc_said startup timed out.' }
        Start-Sleep -Milliseconds 50
    }
    # Windows PowerShell treats native stderr warnings as errors. VTune's
    # exit code, rather than its warning stream, decides collection success.
    $ErrorActionPreference = 'Continue'
    & $Vtune -q -collect hotspots -knob sampling-mode=sw -knob enable-stack-collection=true -knob enable-characterization-insights=false -finalization-mode=none -data-limit=0 -search-dir "$PSScriptRoot/../out/x64-RelWithDebInfo" -result-dir "$runDirectory/vtune_results" -target-pid $runner.Id *> "$runDirectory/collection.log"
    $collectorExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($collectorExit -ne 0) { Write-Warning "VTune collection returned $collectorExit; checking saved data." }
    $runner.WaitForExit()
    if (!(Select-String -LiteralPath $scorePath -Pattern 'reading spoke:' -Quiet)) {
        throw 'Dataset run did not finish: no score tally.'
    }
    # A complete dataset and a readable finalized report are both required.
    # VTune can lose its control connection as the target exits, yet save the
    # full capture. Finalize separately and validate it through the report API.
    $ErrorActionPreference = 'Continue'
    & $Vtune -q -finalize -finalization-mode=full -result-dir "$runDirectory/vtune_results" *> "$runDirectory/finalization.log"
    & $Vtune -q -report summary -result-dir "$runDirectory/vtune_results" -report-output "$runDirectory/summary.txt" 2>> "$runDirectory/finalization.log"
    $reportExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($reportExit -ne 0) { throw "VTune report failed: see $runDirectory/finalization.log" }
    Get-Content -LiteralPath $scorePath
} finally {
    if (!$runner.HasExited) { Stop-Process -Id $runner.Id }
}
