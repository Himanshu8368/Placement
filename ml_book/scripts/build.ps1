param(
    [ValidateSet("html", "pdf", "epub", "all")]
    [string]$To = "html"
)

# Quarto writes progress to stderr; check $LASTEXITCODE instead of stopping on stderr.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Books = Join-Path $Root "books"

if (-not (Get-Command quarto -ErrorAction SilentlyContinue)) {
    Write-Error "Quarto is not on PATH. Install it from https://quarto.org/docs/download/ and rerun."
}

# Use the project virtualenv so executed cells find numpy/sklearn/torch.
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) { $env:QUARTO_PYTHON = $VenvPython }

$Keep = Join-Path $Books "_dist"
New-Item -ItemType Directory -Force -Path $Keep | Out-Null

function Save-Sidecars {
    $build = Join-Path $Books "_build"
    if (-not (Test-Path $build)) { return }
    Get-ChildItem $build -Recurse -File |
        Where-Object { $_.Extension -in ".pdf", ".epub" } |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $Keep $_.Name) -Force }
}

function Restore-Sidecars {
    $build = Join-Path $Books "_build"
    Get-ChildItem $Keep -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $build $_.Name) -Force }
}

Push-Location $Books
try {
    if ($To -eq "all") {
        # Each render clears _build, so HTML goes last and the PDF and EPUB are copied back beside it.
        quarto render --to pdf
        Save-Sidecars
        quarto render --to epub
        Save-Sidecars
        quarto render --to html
        Restore-Sidecars
        Write-Output "ALL_FORMATS_READY"
    } else {
        quarto render --to $To
        if ($LASTEXITCODE -ne 0) { throw "quarto render failed ($LASTEXITCODE)" }
        if ($To -ne "html") { Save-Sidecars }
    }
} finally {
    Pop-Location
}
