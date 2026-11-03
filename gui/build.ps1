# 构建 CadensView GUI。Rust 用的是 GNU 工具链，需要 MinGW 的 dlltool 在 PATH 上。
$ErrorActionPreference = 'Stop'
$mingw = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter 'BrechtSanders.WinLibs.POSIX.UCRT*' |
    ForEach-Object { Join-Path $_.FullName 'mingw64\bin' } | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($mingw) { $env:PATH = "$mingw;$env:PATH" }
$cargo = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path $cargo) { $env:PATH = "$cargo;$env:PATH" }
Push-Location $PSScriptRoot
try { cargo build @args } finally { Pop-Location }
