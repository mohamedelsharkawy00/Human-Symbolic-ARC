# The project root is one level up from where this script is located
$rootPath = Split-Path $PSScriptRoot -Parent

# Ask git which files belong to the project rather than keeping a skip list in
# this script: --exclude-standard applies .gitignore, so the two cannot drift
# apart. --cached is everything tracked and --others adds files git has not been
# told to ignore yet. quotepath=false keeps unusual paths readable. The tests
# folder is tracked but it is fixture data, not code, so it is excluded here.
$relPaths = & git -c core.quotepath=false -C $rootPath ls-files --cached --others --exclude-standard -- . ':(exclude)tests'
if (-not $relPaths) {
    Write-Host "git listed no files under $rootPath - is it a working copy?"
    exit 1
}

# Binary or unreadable files are skipped rather than reported
$ErrorActionPreference = "SilentlyContinue"

$results = foreach ($relPath in $relPaths) {
    # git reports forward slashes; the columns below are written Windows style
    $relPath = $relPath.Replace("/", "\")
    $fullPath = Join-Path $rootPath $relPath
    # A tracked file that has been deleted is still listed, so look before reading
    if (-not (Test-Path -LiteralPath $fullPath)) { continue }

    try {
        $lines = (Get-Content -LiteralPath $fullPath | Measure-Object -Line).Lines
        # Directory is the parent path relative to $rootPath
        $directory = [System.IO.Path]::GetDirectoryName($relPath)
        if ([string]::IsNullOrEmpty($directory)) { $directory = "." }

        [PSCustomObject]@{
            LineCount = $lines
            Name = [System.IO.Path]::GetFileName($relPath)
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
