# 构建并启动 CadensView GUI（需仓库根目录有 .venv 与 tools/ffmpeg.exe）。
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build.ps1') @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$exe = Join-Path $PSScriptRoot 'target\debug\cadensview.exe'
Start-Process -FilePath $exe -WorkingDirectory $PSScriptRoot
