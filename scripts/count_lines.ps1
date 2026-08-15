$ErrorActionPreference = "SilentlyContinue"
# The project root is one level up from where this script is located
$rootPath = Split-Path $PSScriptRoot -Parent
$files = Get-ChildItem -Path $rootPath -Recurse -File | Where-Object { $_.FullName -notmatch "\\(\.git|node_modules|out|puzzles)\\" }

$results = foreach ($file in $files) {
    try {
        $lines = (Get-Content $file.FullName | Measure-Object -Line).Lines
        $relPath = $file.FullName.Replace($rootPath + "\", "")
        # Directory is the parent path relative to $rootPath
        $directory = [System.IO.Path]::GetDirectoryName($relPath)
        if ([string]::IsNullOrEmpty($directory)) { $directory = "." }
        
        [PSCustomObject]@{
            LineCount = $lines
            Name = $file.Name
            Directory = $directory
        }
    } catch {
        # Skip files that can't be read (e.g. binary)
    }
}

$sortedResults = $results | Sort-Object LineCount -Descending

if ($sortedResults.Count -gt 0) {
    # Dynamically find max widths
    $maxCountLen = 0
    $maxNameLen = 0
    foreach ($r in $sortedResults) {
        $cLen = ([string]$r.LineCount).Length
        $nLen = ([string]$r.Name).Length
        if ($cLen -gt $maxCountLen) { $maxCountLen = $cLen }
        if ($nLen -gt $maxNameLen) { $maxNameLen = $nLen }
    }
    
    $sortedResults | ForEach-Object {
        $count = [string]$_.LineCount
        $name = [string]$_.Name
        $dir = [string]$_.Directory
        
        # Buffer columns with 4 spaces for readability
        "$($count.PadRight($maxCountLen + 4))$($name.PadRight($maxNameLen + 4))$($dir)"
    } | Set-Content -Path "$PSScriptRoot\line_counts.txt"
}
