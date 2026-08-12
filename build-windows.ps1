[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'Release',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$workspace = $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Visual Studio Installer (vswhere.exe) was not found.'
}
$visualStudio = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $visualStudio) {
    throw 'Visual Studio with the Desktop development with C++ workload was not found.'
}
$cmake = Join-Path $visualStudio 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$vcpkg = Join-Path $visualStudio 'VC\vcpkg\vcpkg.exe'
$toolchain = Join-Path $visualStudio 'VC\vcpkg\scripts\buildsystems\vcpkg.cmake'
$buildDirectory = Join-Path $workspace 'build-msvc'
$installDirectory = Join-Path $workspace 'artifacts'
$installedDependencies = Join-Path $workspace '.vcpkg_installed'

if ($Clean -and (Test-Path -LiteralPath $buildDirectory)) {
    $resolvedBuild = (Resolve-Path -LiteralPath $buildDirectory).Path
    $resolvedWorkspace = (Resolve-Path -LiteralPath $workspace).Path
    if (-not $resolvedBuild.StartsWith($resolvedWorkspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove build directory outside workspace: $resolvedBuild"
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}

& $vcpkg install --triplet x64-windows "--x-install-root=$installedDependencies"
if ($LASTEXITCODE -ne 0) { throw 'vcpkg dependency installation failed.' }

& $cmake -S $workspace -B $buildDirectory -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DVCPKG_INSTALLED_DIR=$installedDependencies" `
    -DBUILD_TESTING=ON
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }

& $cmake --build $buildDirectory --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }

& $cmake --build $buildDirectory --config $Configuration --target RUN_TESTS
if ($LASTEXITCODE -ne 0) { throw 'One or more native tests failed.' }

& $cmake --install $buildDirectory --config $Configuration --prefix $installDirectory
if ($LASTEXITCODE -ne 0) { throw 'Installation failed.' }

Write-Host "VC-2 binaries and libraries are in $installDirectory"
