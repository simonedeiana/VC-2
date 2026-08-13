# bench.ps1 - reproducible single-thread VC-2 HQ benchmark harness.
#
# Methodology: median of N runs, process pinned to one logical CPU at
# AboveNormal priority (reduces scheduler/background noise), output writes
# disabled, fixed workload. Run from the workspace root.
#
# Examples:
#   .\bench\bench.ps1 -Mode encoder
#   .\bench\bench.ps1 -Mode decoder
#   .\bench\bench.ps1 -Mode all -Runs 9
#   .\bench\bench.ps1 -Mode encode-stream   # (re)generate the decoder input stream
#   .\bench\bench.ps1 -Mode compare -Base bench\baseline\vc2encode.exe   # A/B encoder

param(
  [ValidateSet('encoder', 'decoder', 'all', 'encode-stream', 'compare')]
  [string]$Mode = 'all',
  [int]$Frames = 60,
  [int]$Threads = 1,
  [int]$Runs = 7,
  [string]$EncoderExe = 'build-msvc\bin\vc2encode.exe',
  [string]$DecoderExe = 'build-msvc\bin\vc2decode.exe',
  [string]$InputRaw = 'picture_0.raw',
  [string]$Stream = 'bench\enc-single.vc2',
  [string]$Affinity = '0x100',
  [string]$Priority = 'AboveNormal',
  [string]$Base = ''
)

$ErrorActionPreference = 'Continue'   # PS 5.1: native stderr would otherwise surface as errors

function Invoke-Pinned([string]$Exe, [string[]]$ArgList, [string]$OutFile, [string]$Aff, [string]$Pri) {
  $p = Start-Process -FilePath $Exe -ArgumentList $ArgList -NoNewWindow -PassThru `
       -RedirectStandardOutput $OutFile -RedirectStandardError "$OutFile.err"
  try { $p.ProcessorAffinity = [long]$Aff } catch { }
  try { $p.PriorityClass = $Pri } catch { }
  $p.WaitForExit()
  $p.Refresh()
  return [int]$p.ExitCode
}

function Parse-EncoderFps([string]$Content) {
  if ($Content -match 'Encoded \d+ frames in [\d.]+s -- ([\d.]+) fps') { return [double]$Matches[1] }
  throw "Could not parse encoder output: $Content"
}

function Parse-DecoderFps([string]$Content) {
  # fps line is printed as "  %5.3ffps" (e.g. "  50.186fps"); match digits immediately
  # followed by "fps". "Frame Rate : 25 fps" has a space before fps so it won't match.
  if ($Content -match '([\d.]+)fps') { return [double]$Matches[1] }
  throw "Could not parse decoder output: $Content"
}

function Median([double[]]$Values) {
  $sorted = $Values | Sort-Object
  $n = $sorted.Count
  if ($n -eq 0) { return 0.0 }
  $mid = [int][Math]::Floor($n / 2)
  if ($n % 2 -eq 1) { return $sorted[$mid] }
  return ($sorted[$mid - 1] + $sorted[$mid]) / 2.0
}

$workspace = Split-Path -Parent $PSScriptRoot
Push-Location $workspace
try {
  $enc = Join-Path $workspace $EncoderExe
  $dec = Join-Path $workspace $DecoderExe
  if (-not (Test-Path $enc)) { throw "Encoder not found: $enc" }
  if (-not (Test-Path $dec)) { throw "Decoder not found: $dec" }
  $null = New-Item -ItemType Directory -Force -Path (Join-Path $workspace 'bench')

  $commonArgs = @('--speed=fastest', '--wavelet=haar0', '--depth=3', "--threads=$Threads", "--num-frames=$Frames", '--ratio=2')
  $encArgs = $commonArgs + @('--disable-output', $InputRaw)

  if ($Mode -eq 'encode-stream' -or $Mode -eq 'all') {
    Write-Host "Generating single-thread stream: $Stream"
    $null = Invoke-Pinned $enc @($commonArgs + @($InputRaw, $Stream)) 'bench\_gen.out' $Affinity $Priority
    if (-not (Test-Path $Stream)) { throw "Stream generation failed (no output file)." }
  }

  if ($Mode -eq 'encoder' -or $Mode -eq 'all') {
    $vals = @()
    for ($i = 0; $i -lt $Runs; $i++) {
      $null = Invoke-Pinned $enc $encArgs "bench\_enc$i.out" $Affinity $Priority
      $vals += Parse-EncoderFps (Get-Content "bench\_enc$i.out" -Raw)
    }
    Write-Host ("encoder: runs={0} median={1:F3} fps  all=[{2}]" -f $Runs, (Median $vals), ($vals -join ', '))
  }

  if ($Mode -eq 'decoder' -or $Mode -eq 'all') {
    if (-not (Test-Path $Stream)) { throw "Decoder stream missing: $Stream (run -Mode encode-stream first)" }
    $decArgs = @("--threads=$Threads", "--num-frames=$Frames", '--disable-output', $Stream)
    $vals = @()
    for ($i = 0; $i -lt $Runs; $i++) {
      $null = Invoke-Pinned $dec $decArgs "bench\_dec$i.out" $Affinity $Priority
      $vals += Parse-DecoderFps (Get-Content "bench\_dec$i.out" -Raw)
    }
    Write-Host ("decoder: runs={0} median={1:F3} fps  all=[{2}]" -f $Runs, (Median $vals), ($vals -join ', '))
  }

  if ($Mode -eq 'compare') {
    if (-not $Base -or -not (Test-Path $Base)) { throw "A/B comparison needs -Base <path to baseline exe>" }
    $b = @(); $n = @()
    for ($i = 0; $i -lt $Runs; $i++) {
      $null = Invoke-Pinned $Base $encArgs "bench\_cmpB$i.out" $Affinity $Priority
      $b += Parse-EncoderFps (Get-Content "bench\_cmpB$i.out" -Raw)
      $null = Invoke-Pinned $enc $encArgs "bench\_cmpN$i.out" $Affinity $Priority
      $n += Parse-EncoderFps (Get-Content "bench\_cmpN$i.out" -Raw)
    }
    $bm = Median $b; $nm = Median $n
    $pct = 100.0 * ($nm - $bm) / $bm
    Write-Host ("A/B encoder: baseline med={0:F3}  new med={1:F3}  delta={2:+.2f}%" -f $bm, $nm, $pct)
    Write-Host ("  baseline runs=[{0}]" -f ($b -join ', '))
    Write-Host ("  new      runs=[{0}]" -f ($n -join ', '))
  }
}
finally {
  Pop-Location
}
