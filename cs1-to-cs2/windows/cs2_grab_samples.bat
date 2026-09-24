<# : batch portion
@echo off
echo.
echo  CS2 sample grabber - copies a few small reference files (plain .Texture,
echo  .Surface, .Geometry, .Prefab) into cs2_samples.zip on your Desktop.
echo  It only READS your game folders; nothing in them is changed.
echo.
set "GRAB_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "iex (Get-Content -LiteralPath $env:GRAB_SELF -Raw)"
echo.
pause
exit /b
#>

$ErrorActionPreference = 'SilentlyContinue'
$ud    = Join-Path $env:USERPROFILE 'AppData\LocalLow\Colossal Order\Cities Skylines II'
$desk  = [Environment]::GetFolderPath('Desktop')
$stage = Join-Path $desk 'cs2_samples'
$zip   = Join-Path $desk 'cs2_samples.zip'
$capMB = 60
$script:used = 0

Remove-Item -LiteralPath $stage -Recurse -Force
Remove-Item -LiteralPath $zip -Force
New-Item -ItemType Directory -Path $stage | Out-Null
$list = New-Object System.Collections.Generic.List[string]

function Rel($p) { $p.Substring($ud.TrimEnd('\').Length + 1) }

function Grab($file) {
    # copy one file into the staging folder, keeping its path relative to the CS2 folder
    if (-not $file) { return }
    if (($script:used + $file.Length) / 1MB -gt $capMB) { $list.Add("SKIPPED (size cap): " + (Rel $file.FullName)); return }
    $dest = Join-Path $stage (Rel $file.FullName)
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
    $script:used += $file.Length
}

function Listing($dir) {
    $list.Add('')
    $list.Add('### ' + $dir)
    if (-not (Test-Path -LiteralPath $dir)) { $list.Add('(not found)'); return }
    Get-ChildItem -LiteralPath $dir -Recurse -Force | Select-Object -First 3000 | ForEach-Object {
        if ($_.PSIsContainer) { $list.Add('[dir]            ' + (Rel $_.FullName)) }
        else { $list.Add(('{0,14:N0}  {1}' -f $_.Length, (Rel $_.FullName))) }
    }
}

Write-Host ' Listing folders...'
$imported = Join-Path $ud 'ImportedData'
$pylonMod = Get-ChildItem -LiteralPath (Join-Path $ud '.cache\Mods\pdx_mods') -Directory -Filter '115701_*' |
            Sort-Object Name -Descending | Select-Object -First 1
Listing $imported
Listing (Join-Path $ud 'Mods')
Listing (Join-Path $ud '.cache\Mods\local')
Listing (Join-Path $ud '.packages')
if ($pylonMod) { Listing $pylonMod.FullName }

Write-Host ' Copying samples...'
# 1. The mod that ships loose .Texture files: everything except big binaries
if ($pylonMod) {
    Get-ChildItem -LiteralPath $pylonMod.FullName -Recurse -File -Force |
        Where-Object { $_.Extension -notin '.dll', '.pdb' } | Sort-Object Length |
        ForEach-Object { Grab $_ }
}

# 2. ImportedData: every small file (prefabs, geometry, surfaces, ids)...
if (Test-Path -LiteralPath $imported) {
    $all = Get-ChildItem -LiteralPath $imported -Recurse -File -Force
    $all | Where-Object { $_.Extension -ne '.Texture' -and $_.Length -lt 2MB } | ForEach-Object { Grab $_ }
    # ...plus the 8 smallest plain textures and their .cid files
    $all | Where-Object { $_.Extension -eq '.Texture' } | Sort-Object Length | Select-Object -First 8 | ForEach-Object {
        Grab $_
        Grab (Get-Item -LiteralPath ($_.FullName + '.cid'))
    }
}

$list.Insert(0, ('Copied {0:N1} MB (cap {1} MB). Game folder: {2}' -f ($script:used / 1MB), $capMB, $ud))
$list | Set-Content -LiteralPath (Join-Path $stage 'listing.txt') -Encoding UTF8

Write-Host ' Zipping...'
Compress-Archive -LiteralPath (Get-ChildItem -LiteralPath $stage -Force | ForEach-Object FullName) -DestinationPath $zip -Force
Remove-Item -LiteralPath $stage -Recurse -Force

Write-Host ''
Write-Host (' Done: {0}  ({1:N1} MB)' -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB))
Write-Host ' Upload that zip to Claude.'
Start-Process explorer.exe -ArgumentList ('/select,"' + $zip + '"')
