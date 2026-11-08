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

# Use the project virtualenv (or the ML book one) so executed cells find Jupyter.
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = Join-Path (Split-Path -Parent $Root) "ml_book\.venv\Scripts\python.exe" }
if (Test-Path $VenvPython) { $env:QUARTO_PYTHON = $VenvPython } else { $VenvPython = "python" }

# Sync the sidebar, the outline status and the problem index with the chapters on disk.
& $VenvPython (Join-Path $Root "scripts\gen_book.py")
if ($LASTEXITCODE -ne 0) { throw "gen_book.py failed ($LASTEXITCODE)" }

# Structural lint: block the build on dead local paths or unbalanced fenced divs.
# (Missing/hyperlink warnings still print but do not stop the build.)
& $VenvPython (Join-Path $Root "scripts\check_style.py")
if ($LASTEXITCODE -ne 0) { throw "check_style.py found blocking errors ($LASTEXITCODE)" }

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
        & $VenvPython (Join-Path $Root "scripts\fix_search.py")
        Restore-Sidecars
        Write-Output "ALL_FORMATS_READY"
    } else {
        quarto render --to $To
        if ($LASTEXITCODE -ne 0) { throw "quarto render failed ($LASTEXITCODE)" }
        if ($To -eq "html") { & $VenvPython (Join-Path $Root "scripts\fix_search.py") }
        if ($To -ne "html") { Save-Sidecars }
    }
} finally {
    Pop-Location
}
