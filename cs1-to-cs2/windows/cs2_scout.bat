<# : batch portion
@echo off
echo.
echo  CS2 scout - read-only. Looks around your Cities: Skylines II folders
echo  and writes a report to your Desktop. It does not change or delete anything.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$f='%~f0'; iex ((Get-Content -LiteralPath $f -Raw) -replace '(?s)^.*?#>','')"
echo.
pause
exit /b
#>

$ErrorActionPreference = 'SilentlyContinue'
$out = New-Object System.Collections.Generic.List[string]
function Say($s) { $out.Add([string]$s) }
function Head($s) { Say ''; Say ('=' * 70); Say $s; Say ('=' * 70) }
function MB($bytes) { '{0:N1} MB' -f ($bytes / 1MB) }

$ud  = Join-Path $env:USERPROFILE 'AppData\LocalLow\Colossal Order\Cities Skylines II'
$ref = 'D:\modern central\cs2 references'

Head 'CS2 scout report'
Say ("Generated: " + (Get-Date -Format 'yyyy-MM-dd HH:mm'))
Say ("Windows:   " + [Environment]::OSVersion.VersionString)
Say ("User data: $ud  (exists: " + (Test-Path $ud) + ")")

Head '1. Top-level folders in the CS2 user data folder'
if (Test-Path $ud) {
    Get-ChildItem -LiteralPath $ud -Force | Sort-Object Name | ForEach-Object {
        if ($_.PSIsContainer) {
            $files = Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force
            $size = ($files | Measure-Object Length -Sum).Sum
            Say ('[dir]  {0,-40} {1,7} files  {2}' -f $_.Name, @($files).Count, (MB $size))
        } else {
            Say ('[file] {0,-40} {1}' -f $_.Name, (MB $_.Length))
        }
    }
}

Head '2. Folder tree (folders only, 4 levels, Saves/Screenshots/Logs skipped)'
if (Test-Path $ud) {
    $skip = '\\(Saves|Screenshots|Logs|Crashes)(\\|$)'
    $base = $ud.TrimEnd('\').Length + 1
    Get-ChildItem -LiteralPath $ud -Recurse -Directory -Force -Depth 3 |
        Where-Object { $_.FullName -notmatch $skip } |
        Select-Object -First 400 |
        ForEach-Object {
            $rel = $_.FullName.Substring($base)
            $depth = ($rel.Split('\').Count - 1)
            Say (('  ' * $depth) + $_.Name)
        }
}

Head '3. Asset package files (.cok) under the CS2 user data folder'
if (Test-Path $ud) {
    Get-ChildItem -LiteralPath $ud -Recurse -File -Force -Filter *.cok |
        Select-Object -First 300 |
        ForEach-Object { Say ('{0,10}  {1}' -f (MB $_.Length), $_.FullName.Replace($ud, '<CS2>')) }
}

Head '4. Where do loose .Prefab files live? (folders with the most .Prefab files)'
if (Test-Path $ud) {
    Get-ChildItem -LiteralPath $ud -Recurse -File -Force -Filter *.Prefab |
        Group-Object DirectoryName | Sort-Object Count -Descending | Select-Object -First 25 |
        ForEach-Object { Say ('{0,6} prefabs  {1}' -f $_.Count, $_.Name.Replace($ud, '<CS2>')) }
}

Head '5. The Caltrain KISS reference asset - where the game keeps it'
if (Test-Path $ud) {
    Get-ChildItem -LiteralPath $ud -Recurse -Force -Filter 'kisscaltrain01*' |
        Select-Object -First 20 | ForEach-Object { Say $_.FullName.Replace($ud, '<CS2>') }
}

Head '6. Game version and asset/mod errors from Player.log'
$log = Join-Path $ud 'Player.log'
if (Test-Path $log) {
    Say ("Player.log last written: " + (Get-Item $log).LastWriteTime)
    Select-String -LiteralPath $log -Pattern 'version' | Select-Object -First 8 | ForEach-Object { Say ('  ' + $_.Line.Trim()) }
    Say '  --- lines mentioning asset/prefab/mod errors (first 40) ---'
    Select-String -LiteralPath $log -Pattern '(error|exception|failed).*(asset|prefab|mod|cok)|(asset|prefab|mod|cok).*(error|exception|failed)' |
        Select-Object -First 40 | ForEach-Object { Say ('  ' + $_.Line.Trim()) }
} else { Say 'Player.log not found.' }

Head '7. Game install location (Steam)'
$steam = (Get-ItemProperty 'HKCU:\Software\Valve\Steam').SteamPath
Say ("Steam path: $steam")
$libs = @()
if ($steam) {
    $libs += $steam
    $vdf = Join-Path $steam 'steamapps\libraryfolders.vdf'
    if (Test-Path $vdf) {
        Select-String -LiteralPath $vdf -Pattern '"path"\s+"(.+)"' | ForEach-Object {
            $libs += ($_.Matches[0].Groups[1].Value -replace '\\\\', '\')
        }
    }
}
foreach ($l in ($libs | Select-Object -Unique)) {
    $g = Join-Path $l 'steamapps\common\Cities Skylines II'
    if (Test-Path $g) {
        Say ("FOUND: $g")
        Get-ChildItem -LiteralPath $g -Force | ForEach-Object { Say ('   ' + $_.Name) }
    }
}

Head '8. Blender (for automated mesh conversion later)'
$bl = @()
$cmd = Get-Command blender -ErrorAction SilentlyContinue
if ($cmd) { $bl += $cmd.Source }
$bl += Get-ChildItem 'C:\Program Files\Blender Foundation' -Recurse -Filter blender.exe -Depth 2 | ForEach-Object FullName
$bl += Get-ChildItem (Join-Path $steam 'steamapps\common\Blender') -Filter blender.exe | ForEach-Object FullName
if ($bl.Count) { $bl | Select-Object -Unique | ForEach-Object { Say "FOUND: $_" } } else { Say 'Blender not found.' }

Head "9. Your reference folder: $ref"
if (Test-Path -LiteralPath $ref) {
    Get-ChildItem -LiteralPath $ref -Recurse -Force -Depth 2 |
        Select-Object -First 200 |
        ForEach-Object {
            $kind = if ($_.PSIsContainer) { '[dir] ' } else { '{0,10}' -f (MB $_.Length) }
            Say ("$kind  " + $_.FullName.Replace($ref, '<refs>'))
        }
} else { Say 'Not found.' }

$desk = [Environment]::GetFolderPath('Desktop')
$report = Join-Path $desk 'cs2_scout_report.txt'
$out | Set-Content -LiteralPath $report -Encoding UTF8
Write-Host ""
Write-Host " Report written to: $report"
Write-Host " Opening it in Notepad - send that file back to Claude."
Start-Process notepad.exe -ArgumentList ('"' + $report + '"')
