# ============================================================
# organize_datn.ps1
# Don dep thu muc DATN truoc khi nop
# ============================================================

$base = "C:\Users\Kien\Downloads\DATA DATN\Fine tune"
$mbti = "$base\mbti new"
$essay = "$base\Essay"

Write-Host "`n=== 1. Sap xep thu muc MBTI ===" -ForegroundColor Cyan

# Doi ten cac thu muc (fix khoang trang + typo sfft->sft)
$renames = @{
    "$mbti\mistral sfft" = "$mbti\mistral_sft"
    "$mbti\mistral ift"  = "$mbti\mistral_ift"
    "$mbti\qwen sft"     = "$mbti\qwen_sft"
    "$mbti\qwen ift"     = "$mbti\qwen_ift"
}
foreach ($old in $renames.Keys) {
    $new = $renames[$old]
    if (Test-Path $old) {
        Rename-Item $old $new -Force
        Write-Host "  Renamed: $(Split-Path $old -Leaf)  ->  $(Split-Path $new -Leaf)" -ForegroundColor Green
    }
}

# Xoa desktop.ini rac
$iniFiles = Get-ChildItem "$mbti" -Recurse -Filter "desktop.ini" -Force
foreach ($f in $iniFiles) {
    Remove-Item $f.FullName -Force
    Write-Host "  Removed: $($f.FullName)" -ForegroundColor Yellow
}

# Fix qwen_ift: co ca confusion_matrix.png va qwen_ift_cm.png (trung lap)
$qwenIft = "$mbti\qwen_ift"
if ((Test-Path "$qwenIft\confusion_matrix.png") -and (Test-Path "$qwenIft\qwen_ift_cm.png")) {
    Remove-Item "$qwenIft\confusion_matrix.png" -Force
    Write-Host "  Removed duplicate: qwen_ift\confusion_matrix.png" -ForegroundColor Yellow
}

Write-Host "`n=== 2. Sap xep thu muc Essay ===" -ForegroundColor Cyan

# Tao thu muc moi
foreach ($d in @("mistral_sft","mistral_ift","qwen_sft","qwen_ift")) {
    $path = "$essay\$d"
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "  Created: $d\" -ForegroundColor Green
    }
}

# --- Mistral SFT ---
$mitralSrc = "$essay\Mitral"
$mitralSFTSrc = "$essay\Mitral SFT"
# CSV files (tu Mitral/)
@(
    @{ src="$mitralSrc\summary_mistral_sft_essays.csv";      dst="$essay\mistral_sft\summary_mistral_sft.csv" }
    @{ src="$mitralSrc\predictions_mistral_sft_essays.csv";  dst="$essay\mistral_sft\predictions_mistral_sft.csv" }
    @{ src="$mitralSrc\training_logs_mistral_sft_essays.csv";dst="$essay\mistral_sft\training_logs_mistral_sft.csv" }
    @{ src="$mitralSrc\test_essays_mistral_ sft.csv";        dst="$essay\mistral_sft\test_essays_mistral_sft.csv" }
) | ForEach-Object {
    if (Test-Path $_.src) {
        Copy-Item $_.src $_.dst -Force
        Write-Host "  Copied: $(Split-Path $_.src -Leaf)  ->  mistral_sft\$(Split-Path $_.dst -Leaf)" -ForegroundColor Green
    }
}
# PNG images (tu Mitral SFT/)
if (Test-Path $mitralSFTSrc) {
    Get-ChildItem $mitralSFTSrc -Filter "*.png" | ForEach-Object {
        Copy-Item $_.FullName "$essay\mistral_sft\$($_.Name)" -Force
        Write-Host "  Copied: $($_.Name)  ->  mistral_sft\" -ForegroundColor Green
    }
}

# --- Mistral IFT ---
$mitralIFTSrc = "$essay\Mitral IFT"
@(
    @{ src="$mitralSrc\summary_mistral_ift_essays.csv";      dst="$essay\mistral_ift\summary_mistral_ift.csv" }
    @{ src="$mitralSrc\predictions_mistral_ift_essays.csv";  dst="$essay\mistral_ift\predictions_mistral_ift.csv" }
    @{ src="$mitralSrc\training_logs_mistral_ift_essays.csv";dst="$essay\mistral_ift\training_logs_mistral_ift.csv" }
    @{ src="$mitralSrc\test_essays_fixed (1) ift mistral.csv"; dst="$essay\mistral_ift\test_essays_mistral_ift.csv" }
) | ForEach-Object {
    if (Test-Path $_.src) {
        Copy-Item $_.src $_.dst -Force
        Write-Host "  Copied: $(Split-Path $_.src -Leaf)  ->  mistral_ift\$(Split-Path $_.dst -Leaf)" -ForegroundColor Green
    }
}
if (Test-Path $mitralIFTSrc) {
    Get-ChildItem $mitralIFTSrc -Filter "*.png" | ForEach-Object {
        Copy-Item $_.FullName "$essay\mistral_ift\$($_.Name)" -Force
        Write-Host "  Copied: $($_.Name)  ->  mistral_ift\" -ForegroundColor Green
    }
}

# --- Qwen SFT ---
$qwenSrc = "$essay\Qwen"
$qwenSFTImg = "$essay\Qwen SFT"
@(
    @{ src="$qwenSrc\summary_sft_essays (1).csv";  dst="$essay\qwen_sft\summary_qwen_sft.csv" }
    @{ src="$qwenSrc\training_logs (1).csv";        dst="$essay\qwen_sft\training_logs_qwen_sft.csv" }
    @{ src="$qwenSrc\predictions_sft_essays.csv";   dst="$essay\qwen_sft\predictions_qwen_sft.csv" }
    @{ src="$qwenSrc\test_essays.csv";              dst="$essay\qwen_sft\test_essays_qwen_sft.csv" }
) | ForEach-Object {
    if (Test-Path $_.src) {
        Copy-Item $_.src $_.dst -Force
        Write-Host "  Copied: $(Split-Path $_.src -Leaf)  ->  qwen_sft\$(Split-Path $_.dst -Leaf)" -ForegroundColor Green
    }
}
if (Test-Path $qwenSFTImg) {
    Get-ChildItem $qwenSFTImg -Filter "*.png" | ForEach-Object {
        Copy-Item $_.FullName "$essay\qwen_sft\$($_.Name)" -Force
        Write-Host "  Copied: $($_.Name)  ->  qwen_sft\" -ForegroundColor Green
    }
}

# --- Qwen IFT ---
$qwenIFTImg = "$essay\Qwen IFT"
@(
    @{ src="$qwenSrc\summary_qwen_ift_essays.csv";      dst="$essay\qwen_ift\summary_qwen_ift.csv" }
    @{ src="$qwenSrc\predictions_qwen_ift_essays.csv";  dst="$essay\qwen_ift\predictions_qwen_ift.csv" }
    @{ src="$qwenSrc\test_essays _qwen_ift.csv";        dst="$essay\qwen_ift\test_essays_qwen_ift.csv" }
    @{ src="$qwenSrc\training_logs _qwen_ift.csv";      dst="$essay\qwen_ift\training_logs_qwen_ift.csv" }
) | ForEach-Object {
    if (Test-Path $_.src) {
        Copy-Item $_.src $_.dst -Force
        Write-Host "  Copied: $(Split-Path $_.src -Leaf)  ->  qwen_ift\$(Split-Path $_.dst -Leaf)" -ForegroundColor Green
    }
}
if (Test-Path $qwenIFTImg) {
    Get-ChildItem $qwenIFTImg -Filter "*.png" | ForEach-Object {
        Copy-Item $_.FullName "$essay\qwen_ift\$($_.Name)" -Force
        Write-Host "  Copied: $($_.Name)  ->  qwen_ift\" -ForegroundColor Green
    }
}

# Xoa cac thu muc cu (Mitral, Mitral SFT, Mitral IFT, Qwen, Qwen SFT, Qwen IFT)
$oldDirs = @("Mitral","Mitral SFT","Mitral IFT","Qwen","Qwen SFT","Qwen IFT")
foreach ($d in $oldDirs) {
    $p = "$essay\$d"
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force
        Write-Host "  Removed old folder: $d\" -ForegroundColor Yellow
    }
}

Write-Host "`n=== HOAN THANH! ===" -ForegroundColor Cyan
Write-Host "Ket qua:"
Write-Host "  mbti new\  -> mistral_sft\, mistral_ift\, qwen_sft\, qwen_ift\"
Write-Host "  Essay\     -> mistral_sft\, mistral_ift\, qwen_sft\, qwen_ift\"
Write-Host ""
Read-Host "Nhan Enter de dong"
