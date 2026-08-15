"""Build, validate, and benchmark a VC-2 candidate."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core import load_config, median


ENCODER_FPS_RE = re.compile(r"Encoded\s+\d+\s+frames\s+in\s+[\d.]+s\s+--\s+([\d.]+)\s+fps")
DECODER_FPS_RE = re.compile(r"([\d.]+)fps")


class EvaluationFailure(RuntimeError):
    pass


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update({key: str(value) for key, value in env.items()})
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    process.elapsed_seconds = time.perf_counter() - started  # type: ignore[attr-defined]
    return process


def require_success(process: subprocess.CompletedProcess[str], label: str) -> None:
    if process.returncode:
        tail = "\n".join(process.stdout.splitlines()[-80:])
        raise EvaluationFailure(f"{label} failed with exit code {process.returncode}:\n{tail}")


def git_output(root: Path, *args: str) -> str:
    process = run_command(["git", *args], cwd=root, timeout=60)
    require_success(process, f"git {' '.join(args)}")
    return process.stdout.strip()


def common_checkout_root(root: Path) -> Path:
    common_dir = Path(git_output(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common_dir.parent


def discover_visual_studio() -> dict[str, Path]:
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        raise EvaluationFailure(f"Visual Studio locator not found: {vswhere}")
    process = run_command(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        cwd=Path.cwd(),
        timeout=60,
    )
    require_success(process, "Visual Studio discovery")
    install = Path(process.stdout.strip())
    cmake_bin = install / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin"
    return {
        "cmake": cmake_bin / "cmake.exe",
        "ctest": cmake_bin / "ctest.exe",
        "vcpkg": install / "VC" / "vcpkg" / "vcpkg.exe",
        "toolchain": install / "VC" / "vcpkg" / "scripts" / "buildsystems" / "vcpkg.cmake",
    }


def ensure_dependencies(root: Path, tools: dict[str, Path]) -> Path:
    shared = common_checkout_root(root) / ".vcpkg_installed"
    boost = shared / "x64-windows" / "include" / "boost"
    if boost.is_dir():
        return shared
    process = run_command(
        [str(tools["vcpkg"]), "install", "--triplet", "x64-windows", f"--x-install-root={shared}"],
        cwd=root,
        timeout=3600,
    )
    require_success(process, "vcpkg install")
    return shared


def build(root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, float]]:
    tools = discover_visual_studio()
    installed = ensure_dependencies(root, tools)
    build_config = config.get("build", {})
    build_dir = root / build_config.get("directory", "build-autoresearch")
    configure = [
        str(tools["cmake"]),
        "-S",
        str(root),
        "-B",
        str(build_dir),
        "-A",
        "x64",
        f"-DCMAKE_TOOLCHAIN_FILE={tools['toolchain']}",
        f"-DVCPKG_INSTALLED_DIR={installed}",
        "-DBUILD_TESTING=ON",
        f"-DVC2_ENABLE_CUDA={'ON' if build_config.get('cuda') else 'OFF'}",
    ]
    metrics: dict[str, float] = {}
    process = run_command(configure, cwd=root, timeout=900)
    require_success(process, "CMake configure")
    metrics["configure_seconds"] = process.elapsed_seconds  # type: ignore[attr-defined]

    process = run_command(
        [str(tools["cmake"]), "--build", str(build_dir), "--config", "Release", "--parallel"],
        cwd=root,
        timeout=1800,
    )
    require_success(process, "Release build")
    metrics["build_seconds"] = process.elapsed_seconds  # type: ignore[attr-defined]
    return build_dir, metrics


def run_native_tests(root: Path, build_dir: Path) -> tuple[int, float]:
    tools = discover_visual_studio()
    process = run_command(
        [str(tools["ctest"]), "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"],
        cwd=root,
        timeout=900,
    )
    require_success(process, "native tests")
    match = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", process.stdout)
    count = int(match.group(3)) if match else 0
    return count, process.elapsed_seconds  # type: ignore[attr-defined]


def frame_bytes(width: int, height: int) -> bytes:
    values = array.array("H")
    for y in range(height):
        values.extend(((x * 13 + y * 7 + (x ^ y)) & 1023 for x in range(width)))
    chroma_width = width // 2
    for plane_offset in (341, 683):
        for y in range(height):
            values.extend(((plane_offset + x * 5 + y * 11) & 1023 for x in range(chroma_width)))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def ensure_input(data_dir: Path, width: int, height: int, frames: int) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"input-{width}x{height}-{frames}f.raw"
    expected = width * height * 4 * frames
    if path.is_file() and path.stat().st_size == expected:
        return path
    frame = frame_bytes(width, height)
    with path.open("wb") as handle:
        for _ in range(frames):
            handle.write(frame)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def codec_paths(build_dir: Path) -> tuple[Path, Path]:
    suffix = ".exe" if os.name == "nt" else ""
    encoder = build_dir / "bin" / f"vc2encode{suffix}"
    decoder = build_dir / "bin" / f"vc2decode{suffix}"
    if not encoder.is_file() or not decoder.is_file():
        raise EvaluationFailure(f"codec binaries missing under {build_dir / 'bin'}")
    return encoder, decoder


def encoder_args(wavelet: str, frames: int, raw: Path, output: Path | None = None) -> list[str]:
    args = [
        "--speed=fastest",
        f"--wavelet={wavelet}",
        "--depth=3",
        "--threads=1",
        f"--num-frames={frames}",
        "--ratio=2",
    ]
    if output is None:
        args.append("--disable-output")
        args.append(str(raw))
    else:
        args.extend([str(raw), str(output)])
    return args


def parse_fps(output: str, decoder: bool = False) -> float:
    match = (DECODER_FPS_RE if decoder else ENCODER_FPS_RE).search(output)
    if not match:
        raise EvaluationFailure(f"could not parse {'decoder' if decoder else 'encoder'} fps:\n{output[-2000:]}")
    return float(match.group(1))


def validate_outputs(
    root: Path,
    build_dir: Path,
    config: dict[str, Any],
    raw: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    encoder, decoder = codec_paths(build_dir)
    runtime = root / ".autoresearch" / "artifacts"
    runtime.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    streams: dict[str, str] = {}
    smoke_wavelets = config["evaluation"].get("smoke_wavelets", ["haar0"])
    for variant in config["evaluation"].get("variants", [{"name": "cpu", "environment": {}}]):
        variant_name = variant["name"]
        environment = variant.get("environment", {})
        for wavelet in smoke_wavelets:
            stem = f"{variant_name}-{wavelet}"
            stream = runtime / f"{stem}.vc2"
            decoded = runtime / f"{stem}.yuv"
            process = run_command(
                [str(encoder), *encoder_args(wavelet, 1, raw, stream)],
                cwd=root,
                env=environment,
                timeout=300,
            )
            require_success(process, f"{stem} smoke encode")
            process = run_command(
                [str(decoder), "--threads=1", "--num-frames=1", str(stream), str(decoded)],
                cwd=root,
                env=environment,
                timeout=300,
            )
            require_success(process, f"{stem} smoke decode")
            hashes[f"{stem}_stream_sha256"] = sha256(stream)
            hashes[f"{stem}_pixels_sha256"] = sha256(decoded)
            streams[stem] = str(stream)
    return hashes, streams


def find_validator(root: Path) -> Path | None:
    names = ["vc2-bitstream-validator.exe", "vc2-bitstream-validator"]
    for checkout in (root, common_checkout_root(root)):
        for name in names:
            candidate = checkout / ".venv-validator" / "Scripts" / name
            if candidate.is_file():
                return candidate
    discovered = shutil.which("vc2-bitstream-validator")
    return Path(discovered) if discovered else None


def run_validator(root: Path, streams: dict[str, str]) -> int:
    validator = find_validator(root)
    if validator is None:
        return 0
    validated = 0
    validator_dir = root / ".autoresearch" / "validator"
    validator_dir.mkdir(parents=True, exist_ok=True)
    for stem, stream in streams.items():
        template = validator_dir / f"{stem}-%d.raw"
        process = run_command([str(validator), stream, "--output", str(template)], cwd=root, timeout=600)
        require_success(process, f"validator {stem}")
        if "No errors found in bitstream" not in process.stdout:
            raise EvaluationFailure(f"validator did not report success for {stem}")
        validated += 1
    return validated


def benchmark(
    root: Path,
    build_dir: Path,
    config: dict[str, Any],
    raw: Path,
    frames: int,
    runs: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    encoder, decoder = codec_paths(build_dir)
    runtime = root / ".autoresearch" / "bench"
    runtime.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    wavelets = config["evaluation"].get("wavelets", ["haar0"])
    variants = config["evaluation"].get("variants", [{"name": "cpu", "environment": {}}])
    for variant in variants:
        variant_name = variant["name"]
        environment = variant.get("environment", {})
        for wavelet in wavelets:
            stream = runtime / f"{variant_name}-{wavelet}-{frames}f.vc2"
            process = run_command(
                [str(encoder), *encoder_args(wavelet, frames, raw, stream)],
                cwd=root,
                env=environment,
                timeout=900,
            )
            require_success(process, f"prepare {variant_name} {wavelet} stream")
            encoder_samples: list[float] = []
            decoder_samples: list[float] = []
            for _ in range(runs):
                process = run_command(
                    [str(encoder), *encoder_args(wavelet, frames, raw)],
                    cwd=root,
                    env=environment,
                    timeout=900,
                )
                require_success(process, f"benchmark {variant_name} encoder {wavelet}")
                encoder_samples.append(parse_fps(process.stdout))
                process = run_command(
                    [str(decoder), "--threads=1", f"--num-frames={frames}", "--disable-output", str(stream)],
                    cwd=root,
                    env=environment,
                    timeout=900,
                )
                require_success(process, f"benchmark {variant_name} decoder {wavelet}")
                decoder_samples.append(parse_fps(process.stdout, decoder=True))
            safe_wavelet = wavelet.replace("deslauriers-debuc-", "dd").replace("-", "_")
            enc_key = f"{variant_name}_encoder_{safe_wavelet}_fps"
            dec_key = f"{variant_name}_decoder_{safe_wavelet}_fps"
            metrics[enc_key] = median(encoder_samples)
            metrics[dec_key] = median(decoder_samples)
            samples[enc_key] = encoder_samples
            samples[dec_key] = decoder_samples
    return metrics, samples


def evaluate(root: Path, config: dict[str, Any], tier: str) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "valid": False,
        "tier": tier,
        "branch": config["branch"],
        "metrics": {},
        "samples": {},
        "hashes": {},
        "stages": [],
    }
    try:
        build_dir, build_metrics = build(root, config)
        result["metrics"].update(build_metrics)
        result["stages"].append("build")
        test_count, test_seconds = run_native_tests(root, build_dir)
        result["metrics"].update({"native_tests": test_count, "test_seconds": test_seconds})
        result["stages"].append("native-tests")

        tier_config = config["evaluation"]["tiers"][tier]
        frames = int(tier_config["frames"])
        runs = int(tier_config["runs"])
        data_dir = root / ".autoresearch" / "data"
        smoke_raw = ensure_input(data_dir, int(config["evaluation"]["width"]), int(config["evaluation"]["height"]), 1)
        hashes, streams = validate_outputs(root, build_dir, config, smoke_raw)
        result["hashes"] = hashes
        result["stages"].append("codec-smoke")
        if tier_config.get("validator", False):
            result["metrics"]["validated_streams"] = run_validator(root, streams)
            result["stages"].append("conformance-validator")

        raw = ensure_input(data_dir, int(config["evaluation"]["width"]), int(config["evaluation"]["height"]), frames)
        metrics, samples = benchmark(root, build_dir, config, raw, frames, runs)
        result["metrics"].update(metrics)
        result["samples"] = samples
        result["stages"].append(f"benchmark-{tier}")
        result["valid"] = True
    except (EvaluationFailure, subprocess.TimeoutExpired, OSError, ValueError) as exc:
        result["error"] = str(exc)
    result["metrics"]["evaluation_seconds"] = time.perf_counter() - started
    result["commit"] = git_output(root, "rev-parse", "HEAD")
    result["dirty"] = bool(git_output(root, "status", "--porcelain"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", choices=("smoke", "quick", "full"), default="quick")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    result = evaluate(root, load_config(config_path), args.tier)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
