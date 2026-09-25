<# : batch portion
@echo off
set "CS1PORT_SELF=%~f0"
set "CS1PORT_ARG1=%~1"
set "CS1PORT_ARG2=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "iex (Get-Content -LiteralPath $env:CS1PORT_SELF -Raw)"
echo.
pause
exit /b
#>
# cs2_import.bat: put this next to your CS1-port package zips (SJX40_MU.zip, ...) and double-click it.
#   - every *.zip in this folder is unpacked and installed into the game's ImportedData folder;
#     a package that was installed before is replaced cleanly (its old files are removed first)
#   - packages installed earlier whose zip is no longer here are offered for removal
#   - "cs2_import.bat remove NAME" removes one installed package
$ErrorActionPreference = 'Stop'
$here   = Split-Path -Parent $env:CS1PORT_SELF
$mode   = $env:CS1PORT_ARG1
$arg    = $env:CS1PORT_ARG2
$ud     = Join-Path $env:USERPROFILE 'AppData\LocalLow\Colossal Order\Cities Skylines II'
$imp    = Join-Path $ud 'ImportedData'
$logDir = Join-Path $ud 'ModsData\CS1Port'

function Remove-Package($name) {
    $logFile = Join-Path $logDir ($name + '.installed.txt')
    if (-not (Test-Path -LiteralPath $logFile)) { return 0 }
    $n = 0
    foreach ($f in Get-Content -LiteralPath $logFile) {
        if ($f -and (Test-Path -LiteralPath $f)) { Remove-Item -LiteralPath $f -Force; $n++ }
    }
    Get-ChildItem -LiteralPath $imp -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $name -and @(Get-ChildItem -LiteralPath $_.FullName -Force).Count -eq 0 } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    Remove-Item -LiteralPath $logFile -Force
    return $n
}

Write-Host ""
Write-Host " CS2 import folder: $here"
try {
    if (Get-Process -Name 'Cities2' -ErrorAction SilentlyContinue) { throw 'Cities: Skylines II is running. Close the game first, then run this again.' }
    if (-not (Test-Path -LiteralPath $ud)) { throw "Game data folder not found: $ud" }
    New-Item -ItemType Directory -Path $imp -Force | Out-Null
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    if ($mode -eq 'remove') {
        if (-not $arg) { throw 'Usage: cs2_import.bat remove PACKAGENAME' }
        $n = Remove-Package $arg
        Write-Host " Removed '$arg' ($n files)."
        return
    }
    # the importer keeps assets in ImportedData\<16 hex chars>\ ; use the existing (fullest) one
    $target = Get-ChildItem -LiteralPath $imp -Directory | Where-Object { $_.Name -match '^[0-9a-f]{16}$' } |
              Sort-Object { @(Get-ChildItem -LiteralPath $_.FullName -File).Count } -Descending | Select-Object -First 1
    if ($target) { $target = $target.FullName } else { $target = Join-Path $imp 'c51c0c51c0c51c00'; New-Item -ItemType Directory -Path $target -Force | Out-Null }
    $icons = Join-Path $imp 'TextureAssets'
    New-Item -ItemType Directory -Path $icons -Force | Out-Null

    $zips = @(Get-ChildItem -LiteralPath $here -File -Filter '*.zip')
    if ($zips.Count -eq 0) { Write-Host " No .zip packages next to this script." }
    $installed = @{}
    foreach ($zip in $zips) {
        $name = [IO.Path]::GetFileNameWithoutExtension($zip.Name)
        $work = Join-Path (Join-Path $here '_unpacked') $name
        if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
        Expand-Archive -LiteralPath $zip.FullName -DestinationPath $work -Force
        $payload = Get-ChildItem -LiteralPath $work -Recurse -Directory -Filter 'payload' | Select-Object -First 1
        if (-not $payload) { Write-Host " $($zip.Name): no payload folder inside, skipped."; continue }
        $pkg = Split-Path -Leaf (Split-Path -Parent $payload.FullName)
        $old = Remove-Package $pkg
        $log = New-Object System.Collections.Generic.List[string]
        foreach ($pair in @(@('asset', $target), @('TextureAssets', $icons))) {
            $src = Join-Path $payload.FullName $pair[0]
            if (-not (Test-Path -LiteralPath $src)) { continue }
            Get-ChildItem -LiteralPath $src -Recurse -File | ForEach-Object {
                $rel  = $_.FullName.Substring($src.Length + 1)
                $dest = Join-Path $pair[1] $rel
                New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
                Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
                $log.Add($dest)
            }
        }
        $log | Set-Content -LiteralPath (Join-Path $logDir ($pkg + '.installed.txt')) -Encoding UTF8
        $installed[$pkg] = $true
        $note = ''
        if ($old) { $note = " (replaced the previous version, $old files)" }
        Write-Host (" {0}: {1} files installed{2}" -f $pkg, $log.Count, $note)
    }
    # packages installed by an earlier run whose zip is gone from this folder
    foreach ($logFile in @(Get-ChildItem -LiteralPath $logDir -File -Filter '*.installed.txt')) {
        $pkg = $logFile.Name -replace '\.installed\.txt$', ''
        if ($installed.ContainsKey($pkg)) { continue }
        if ($env:CS1PORT_YES) { $answer = 'y' } else { $answer = Read-Host " '$pkg' is installed but there is no $pkg.zip here. Remove it? [y/N]" }
        if ($answer -match '^[yY]') { $n = Remove-Package $pkg; Write-Host "   removed '$pkg' ($n files)" }
    }
    Write-Host ""
    Write-Host " Done. Start the game."
} catch {
    Write-Host ""
    Write-Host (" FAILED: " + $_.Exception.Message) -ForegroundColor Red
}
