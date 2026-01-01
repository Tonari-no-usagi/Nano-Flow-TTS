# 1. Visual Studio コンパイラ (cl.exe) のパス設定
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if (Test-Path $vsPath) {
    $env:PATH = "$vsPath;" + $env:PATH
}

# 2. 日本語ユーザー名（全角スペース）によるビルドエラー対策
# 短いパス形式 (8.3形式) を使用して、nvccがパスを誤認するのを防ぎます
$env:CARGO_HOME = "C:\Users\6372~1\.cargo"
$env:RUSTUP_HOME = "C:\Users\6372~1\.rustup"
# CUDAが一時ファイル作成時にスペースでこけないよう、一時フォルダも短いパスへ
$env:TMP = "C:\Users\6372~1\AppData\Local\Temp"
$env:TEMP = "C:\Users\6372~1\AppData\Local\Temp"

Write-Host "--- Build environment initialized (Short Path + MSVC) ---" -ForegroundColor Cyan
Write-Host "Starting nflow-train..." -ForegroundColor Green

cargo run --release -p nflow-train -- train --limit-mins 60
