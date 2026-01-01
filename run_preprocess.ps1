# 1. Visual Studio コンパイラ (cl.exe) のパス設定
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if (Test-Path $vsPath) {
    $env:PATH = "$vsPath;" + $env:PATH
    # nvcc が確実に cl.exe を見つけられるように明示的に指定
    $env:NVCC_CCBIN = "$vsPath\cl.exe"
}

# 2. 日本語ユーザー名（全角スペース）によるビルドエラー対策
$env:CARGO_HOME = "C:\Users\6372~1\.cargo"
$env:RUSTUP_HOME = "C:\Users\6372~1\.rustup"
$env:TMP = "C:\Users\6372~1\AppData\Local\Temp"
$env:TEMP = "C:\Users\6372~1\AppData\Local\Temp"

Write-Host "--- Build environment initialized (Preprocess) ---" -ForegroundColor Cyan
Write-Host "Starting mel-spectrogram pre-computation..." -ForegroundColor Green

# preprocessコマンドを実行
cargo run --release -p nflow-train -- preprocess --dir datasets

Write-Host "Pre-computation finished. You can now run 'run_train.ps1'." -ForegroundColor Green