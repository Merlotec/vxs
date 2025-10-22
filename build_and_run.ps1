Param(
  # ========= CONFIG (edit or override via -Param value) =========
  [string]$PyBin      = "python",                       # python executable
  [string]$VenvDir    = ".venv",                        # auto-activate if exists
  [string]$Root       = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [string]$PyDir      = "$((Split-Path -Parent $MyInvocation.MyCommand.Path))\voxelsim-py",
  [string]$RenderDir  = "$((Split-Path -Parent $MyInvocation.MyCommand.Path))\voxelsim-renderer",
  [string]$Features   = $env:FEATURES,                  # leave empty to disable features
  [string]$WheelDir   = "$((Split-Path -Parent $MyInvocation.MyCommand.Path))\voxelsim-py\target\wheels",
  [string]$WorldPort  = $(if ($env:VXS_WORLD_PORT) { $env:VXS_WORLD_PORT } else { "8082" }),
  [string]$AgentPort  = $(if ($env:VXS_AGENT_PORT) { $env:VXS_AGENT_PORT } else { "8083" })
)

# Strict/error settings
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

###############################################################################
# [SECTION A] Optional: activate local venv
# - Comment this whole section if you don't want to auto-activate .venv
###############################################################################
if (Test-Path -LiteralPath "$Root\$VenvDir") {
  Write-Host ">> Activating venv: $Root\$VenvDir"
  # dot-source the Activate script so it affects current session
  $activate = Join-Path "$Root\$VenvDir\Scripts" "Activate.ps1"
  if (Test-Path $activate) {
    . $activate
  } else {
    Write-Warning "Venv activate script not found at $activate"
  }
}

###############################################################################
# [SECTION B] Ensure maturin is available
# - Comment if you are sure maturin is already installed in your env
###############################################################################
if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
  Write-Host ">> maturin not found; installing into current Python: $PyBin"
  & $PyBin -m pip install --upgrade maturin
}

###############################################################################
# [SECTION C] Build voxelsim-py wheel(s) with maturin
# - Comment this whole section if you don’t want to rebuild each time
# - You can change WheelDir above; we pass --out explicitly
###############################################################################
Write-Host ">> Building voxelsim-py with maturin (release, features='$Features')"
New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
Push-Location $PyDir
try {
  $featArgs = @()
  if ($Features) { $featArgs += @("--features", $Features) }
  # Add --no-default-features if you also want to disable Cargo defaults:
  # $featArgs += "--no-default-features"
  maturin build --release @featArgs --out "$WheelDir"
}
finally { Pop-Location }

###############################################################################
# [SECTION D] Install the built wheel(s)
# - Comment this if you don’t want to reinstall each time
###############################################################################
Write-Host ">> Installing wheel(s) from: $WheelDir (force-reinstall)"
$wheels = Get-ChildItem -Path $WheelDir -Filter *.whl -ErrorAction SilentlyContinue
if (-not $wheels) {
  throw "No wheels found in $WheelDir"
}
& $PyBin -m pip install --force-reinstall $wheels.FullName

###############################################################################
# [SECTION E] Run voxelsim-renderer once with default ports
# - This run will block until you stop it (Ctrl-C)
# - Comment this whole section if you want to skip the first run
###############################################################################
Write-Host ">> Running voxelsim-renderer (default ports). Press Ctrl-C to stop."
Push-Location $RenderDir
try {
  cargo run --release
} catch {
  Write-Warning "Renderer exited (default ports): $($_.Exception.Message)"
}
Pop-Location

###############################################################################
# [SECTION F] Run voxelsim-renderer with custom ports
# - Exports env vars and then runs again (blocks until you stop it)
# - Comment if you only want the default run
###############################################################################
Write-Host ">> Running voxelsim-renderer with custom ports:"
Write-Host "   VXS_WORLD_PORT=$WorldPort"
Write-Host "   VXS_AGENT_PORT=$AgentPort"
$env:VXS_WORLD_PORT = $WorldPort
$env:VXS_AGENT_PORT = $AgentPort

Push-Location $RenderDir
cargo run --release
Pop-Location
