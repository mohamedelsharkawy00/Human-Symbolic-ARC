param([string]$Label = "NVIDIA/Intel/OBS DLLs - no PDBs available")

$warnings = 0
$kept = [System.Collections.ArrayList]::new()

foreach ($line in $input) {
    if ($line -match 'Cannot locate (debugging information for )?file') {
        $warnings++
    } elseif ($line.Trim().Length -eq 0) {
        # drop blanks
    } else {
        [void]$kept.Add($line)
    }
}

$kept | ForEach-Object { Write-Host $_ }

if ($warnings -gt 0) {
    Write-Host "[*] $warnings symbol warnings suppressed ($Label)"
}
