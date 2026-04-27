"""test_therock_asan.py — Verify ASAN instrumentation in TheRock build.

Structured pytest tests that verify:
  1. ASAN runtime and build artifacts are present
  2. Key ROCm libraries are ASAN-instrumented
  3. TheRock's clang++ can compile HIP code with -fsanitize=address
  4. ASAN catches intentional host memory errors
  5. ASAN does NOT produce false positives on clean code
  6. HIP runtime APIs work correctly under ASAN (requires GPU)

Usage:
    # Run all (GPU required for some tests):
    python -m pytest test_therock_asan.py -v

    # Run only host-side tests (no GPU required):
    python -m pytest test_therock_asan.py -v -k "not gpu"

    # Run with ASAN output visible:
    python -m pytest test_therock_asan.py -v -s

Environment:
    ROCM_HOME           — ROCm install prefix (default: /opt/rocm)
    ASAN_LIB_DIR        — ASAN overlay lib directory (default: /opt/rocm-asan/lib)
    PYTORCH_ROCM_ARCH   — GPU arch for compilation (default: gfx950)
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

ROCM_HOME = Path(os.getenv("ROCM_HOME", "/opt/rocm"))
ASAN_LIB_DIR = Path(os.getenv("ASAN_LIB_DIR", "/opt/rocm-asan/lib"))
ROCM_ARCH = os.getenv("PYTORCH_ROCM_ARCH", "gfx950")
CLANG = ROCM_HOME / "llvm" / "bin" / "clang++"
THIS_DIR = Path(__file__).resolve().parent


def find_asan_runtime() -> Path | None:
    """Locate libclang_rt.asan-x86_64.so under ROCM_HOME."""
    for p in (ROCM_HOME / "llvm" / "lib" / "clang").rglob(
        "libclang_rt.asan-x86_64.so"
    ):
        return p
    return None


def has_gpu() -> bool:
    """Check whether at least one HIP GPU is visible."""
    try:
        result = subprocess.run(
            [str(ROCM_HOME / "bin" / "rocm_agent_enumerator")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith("gfx"):
                return True
    except Exception:
        pass
    return False


def nm_has_symbol(lib_path: Path, symbol: str) -> bool:
    """Check if a shared library exports a symbol matching *symbol*."""
    try:
        result = subprocess.run(
            ["nm", "-D", str(lib_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return symbol in result.stdout
    except Exception:
        return False


def compile_hip_source(source: str, output: Path, extra_flags: list[str] | None = None) -> str:
    """Compile a HIP C++ source string with ASAN. Returns stderr.

    Uses -shared-libasan so the test binary links against the shared ASAN
    runtime — the same one that TheRock's libraries (libamdhip64.so, etc.)
    were built against.  Without this, the binary gets a static ASAN runtime
    that collides with the shared one at load time:
        "Your application is linked against incompatible ASan runtimes."
    """
    src_file = output.with_suffix(".cpp")
    src_file.write_text(source)
    cmd = [
        str(CLANG),
        "-g",
        "-fsanitize=address",
        "-shared-libasan",
        f"--offload-arch={ROCM_ARCH}",
        "-x", "hip",
        f"-I{ROCM_HOME}/include",
        f"-L{ROCM_HOME}/lib",
        "-lamdhip64",
        "-o", str(output),
        str(src_file),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    logger.info(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result.stderr


def run_asan_binary(binary: Path, args: list[str] | None = None, timeout: int = 60) -> tuple[int, str]:
    """Run an ASAN-instrumented binary. Returns (returncode, combined_output).

    The ASAN runtime is loaded as a NEEDED dependency via LD_LIBRARY_PATH —
    NOT via LD_PRELOAD, which breaks HIP's internal dlopen/dlsym (SEGV in
    amd::Device::init).  verify_asan_link_order=0 suppresses ASAN's check
    that it must be loaded first.

    The existing LD_LIBRARY_PATH ordering is preserved (keeps the ASAN overlay
    dir first); extra LLVM/ROCm runtime dirs are appended only if absent.
    """
    asan_lib = find_asan_runtime()
    cmd = [str(binary)] + (args or [])
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=0:symbolize=1:verify_asan_link_order=0"
    symbolizer = ROCM_HOME / "llvm" / "bin" / "llvm-symbolizer"
    if symbolizer.is_file():
        env["ASAN_SYMBOLIZER_PATH"] = str(symbolizer)

    llvm_lib_dir = str(ROCM_HOME / "llvm" / "lib")
    rocm_lib_dir = str(ROCM_HOME / "lib")
    existing_ldpath = env.get("LD_LIBRARY_PATH", "")
    ld_paths = [p for p in existing_ldpath.split(os.pathsep) if p]
    for extra in (llvm_lib_dir, rocm_lib_dir, str(asan_lib.parent) if asan_lib else None):
        if extra and extra not in ld_paths:
            ld_paths.append(extra)
    if ld_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)

    env.pop("LD_PRELOAD", None)

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=timeout
    )
    combined = result.stdout + "\n" + result.stderr
    return result.returncode, combined


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def asan_runtime():
    """Path to the ASAN runtime .so, or skip if not found."""
    lib = find_asan_runtime()
    if lib is None:
        pytest.skip("ASAN runtime (libclang_rt.asan-x86_64.so) not found")
    return lib


@pytest.fixture(scope="session")
def build_dir():
    """Temporary directory for compiled test binaries."""
    d = tempfile.mkdtemp(prefix="asan_pytest_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def test_hip_binary(build_dir):
    """Compile test_hip_asan.cpp once for the session."""
    src = THIS_DIR / "test_hip_asan.cpp"
    if not src.exists():
        pytest.skip(f"test_hip_asan.cpp not found at {src}")
    output = build_dir / "test_hip_asan"
    compile_hip_source(src.read_text(), output)
    return output


@pytest.fixture(scope="session")
def require_gpu():
    """Skip if no GPU is available."""
    if not has_gpu():
        pytest.skip("No GPU available (need --device=/dev/kfd --device=/dev/dri)")


# ===========================================================================
# Section 1: ASAN Build Artifacts
# ===========================================================================

class TestASANBuildArtifacts:
    """Verify the TheRock build includes ASAN instrumentation."""

    def test_rocm_install_exists(self):
        assert ROCM_HOME.is_dir(), f"ROCM_HOME not found: {ROCM_HOME}"

    def test_clang_available(self):
        assert CLANG.is_file() and os.access(CLANG, os.X_OK), \
            f"clang++ not found or not executable: {CLANG}"

    def test_asan_runtime_exists(self, asan_runtime):
        assert asan_runtime.is_file(), f"ASAN runtime not found: {asan_runtime}"
        size_mb = asan_runtime.stat().st_size / (1024 * 1024)
        logger.info(f"ASAN runtime: {asan_runtime} ({size_mb:.1f} MB)")

    def test_hipcc_available(self):
        hipcc = ROCM_HOME / "bin" / "hipcc"
        assert hipcc.is_file() and os.access(hipcc, os.X_OK), \
            f"hipcc not found: {hipcc}"


class TestASANInstrumentation:
    """Verify that key ROCm libraries contain ASAN symbols.

    With the two-pass overlay architecture, ASAN-instrumented libs live in
    ASAN_LIB_DIR (/opt/rocm-asan/lib), not in ROCM_HOME/lib.  The normal
    libs at ROCM_HOME/lib should NOT have ASAN symbols.
    """

    @pytest.mark.parametrize("lib_name", [
        "libamdhip64.so",
        "libhsa-runtime64.so",
    ])
    def test_overlay_library_has_asan_symbols(self, lib_name):
        lib_path = ASAN_LIB_DIR / lib_name
        if not lib_path.exists():
            # Fall back to ROCM_HOME for single-pass builds
            lib_path = ROCM_HOME / "lib" / lib_name
        if not lib_path.exists():
            pytest.skip(f"{lib_name} not found at {ASAN_LIB_DIR} or {ROCM_HOME / 'lib'}")
        assert nm_has_symbol(lib_path, "__asan_report"), \
            f"{lib_name} at {lib_path} does not contain __asan_report — not ASAN-instrumented"

    @pytest.mark.parametrize("lib_name", [
        "libamdhip64.so",
        "libhsa-runtime64.so",
    ])
    def test_overlay_library_has_asan_init(self, lib_name):
        lib_path = ASAN_LIB_DIR / lib_name
        if not lib_path.exists():
            lib_path = ROCM_HOME / "lib" / lib_name
        if not lib_path.exists():
            pytest.skip(f"{lib_name} not found at {ASAN_LIB_DIR} or {ROCM_HOME / 'lib'}")
        assert nm_has_symbol(lib_path, "__asan_init"), \
            f"{lib_name} at {lib_path} does not contain __asan_init"

    @pytest.mark.parametrize("lib_name", [
        "libamdhip64.so",
        "libhsa-runtime64.so",
    ])
    def test_normal_library_not_asan(self, lib_name):
        """The normal ROCm libs should NOT be ASAN-instrumented."""
        lib_path = ROCM_HOME / "lib" / lib_name
        if not lib_path.exists():
            pytest.skip(f"{lib_name} not found at {lib_path}")
        if not ASAN_LIB_DIR.exists():
            pytest.skip("No overlay directory — single-pass build, skip this check")
        assert not nm_has_symbol(lib_path, "__asan_report"), \
            f"Normal {lib_name} at {lib_path} should NOT be ASAN-instrumented"

    def test_non_asan_compiler(self):
        """TheRock's compiler binary should NOT be directly linked to ASAN.

        Uses readelf to check the binary's NEEDED entries rather than ldd,
        because ldd resolves transitive dependencies and LD_LIBRARY_PATH
        entries, which would cause false positives in the ASAN overlay setup.
        """
        clang_bin = ROCM_HOME / "llvm" / "bin" / "clang"
        if not clang_bin.exists():
            pytest.skip("clang binary not found")
        result = subprocess.run(
            ["readelf", "-d", str(clang_bin)], capture_output=True, text=True
        )
        assert "libclang_rt.asan" not in result.stdout, \
            "The compiler itself should not be directly linked to ASAN runtime"


# ===========================================================================
# Section 2: ASAN Compilation
# ===========================================================================

class TestASANCompilation:
    """Verify that HIP code can be compiled with -fsanitize=address."""

    def test_compile_simple_hip(self, build_dir):
        source = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            #include <cstdio>
            __global__ void k(int* p) { p[threadIdx.x] = threadIdx.x; }
            int main() {
                printf("compiled with ASAN\\n");
                return 0;
            }
        """)
        output = build_dir / "simple_hip"
        compile_hip_source(source, output)
        assert output.is_file() and os.access(output, os.X_OK)

    def test_compiled_binary_has_asan(self, build_dir):
        """A binary compiled with -fsanitize=address should link to ASAN runtime."""
        source = textwrap.dedent("""\
            #include <cstdio>
            int main() { printf("hello\\n"); return 0; }
        """)
        output = build_dir / "asan_linked"
        compile_hip_source(source, output)
        result = subprocess.run(
            ["ldd", str(output)], capture_output=True, text=True
        )
        assert "libclang_rt.asan" in result.stdout, \
            "Binary compiled with -fsanitize=address should link to ASAN runtime"

    def test_compile_test_hip_asan(self, test_hip_binary):
        """The full test_hip_asan.cpp compiles successfully."""
        assert test_hip_binary.is_file()


# ===========================================================================
# Section 3: ASAN Error Detection (host-side, no GPU required)
# ===========================================================================

class TestASANErrorDetection:
    """Verify ASAN catches intentional host-side memory errors."""

    def test_heap_buffer_overflow_detected(self, test_hip_binary):
        rc, output = run_asan_binary(test_hip_binary, ["heap_overflow"])
        assert "heap-buffer-overflow" in output, \
            f"ASAN did not report heap-buffer-overflow:\n{output[-500:]}"

    def test_use_after_free_detected(self, test_hip_binary):
        rc, output = run_asan_binary(test_hip_binary, ["use_after_free"])
        assert "heap-use-after-free" in output or "use-after-free" in output, \
            f"ASAN did not report use-after-free:\n{output[-500:]}"

    def test_asan_reports_have_stack_traces(self, test_hip_binary):
        """ASAN reports should include symbolized stack traces."""
        rc, output = run_asan_binary(test_hip_binary, ["heap_overflow"])
        assert "#0" in output or "#1" in output, \
            f"ASAN report lacks stack traces (symbolize=1 may not be working):\n{output[-500:]}"

    def test_asan_reports_show_source_location(self, test_hip_binary):
        """ASAN reports should reference test_hip_asan.cpp."""
        rc, output = run_asan_binary(test_hip_binary, ["heap_overflow"])
        assert "test_hip_asan.cpp" in output, \
            f"ASAN report doesn't reference source file:\n{output[-500:]}"


class TestASANNoFalsePositives:
    """Verify ASAN does not flag correct host-side code."""

    def test_clean_malloc_free(self, build_dir):
        """A program that does correct malloc/memset/free should produce no ASAN errors."""
        source = textwrap.dedent("""\
            #include <cstdlib>
            #include <cstring>
            #include <cstdio>
            int main() {
                for (int i = 0; i < 100; i++) {
                    void* p = malloc(1024);
                    memset(p, 0, 1024);
                    free(p);
                }
                printf("clean_malloc_free: OK\\n");
                return 0;
            }
        """)
        output = build_dir / "clean_malloc"
        compile_hip_source(source, output)
        rc, out = run_asan_binary(output)
        assert rc == 0, f"Clean program exited with rc={rc}"
        assert "ERROR: AddressSanitizer" not in out, \
            f"ASAN false positive on clean code:\n{out[-500:]}"


# ===========================================================================
# Section 4: HIP Runtime Under ASAN (GPU required)
# ===========================================================================

@pytest.mark.usefixtures("require_gpu")
class TestHIPRuntimeUnderASAN:
    """Verify HIP runtime APIs work correctly under ASAN (requires GPU)."""

    def test_clean_hip_program(self, test_hip_binary):
        """A correct HIP program (vec_add) should have no ASAN errors."""
        rc, output = run_asan_binary(test_hip_binary, ["clean"])
        assert rc == 0, f"Clean HIP program failed (rc={rc}):\n{output[-500:]}"
        assert "ERROR: AddressSanitizer" not in output, \
            f"ASAN false positive in clean HIP program:\n{output[-500:]}"
        assert "correct" in output, \
            f"HIP vec_add computed wrong result:\n{output[-500:]}"

    def test_hip_event_query(self, test_hip_binary):
        """hipEventQuery polling loop should complete without crash."""
        rc, output = run_asan_binary(test_hip_binary, ["event_query"], timeout=120)
        assert rc == 0, f"hipEventQuery stress test failed (rc={rc}):\n{output[-500:]}"
        assert "correct" in output, \
            f"hipEventQuery test computed wrong result:\n{output[-500:]}"

    def test_hip_event_query_asan_findings(self, test_hip_binary):
        """Report (but don't fail on) any ASAN findings in hipEventQuery.

        This is the primary debugging target — if ASAN finds memory errors
        in the HIP runtime's event handling code, we want to see them.
        """
        rc, output = run_asan_binary(test_hip_binary, ["event_query"], timeout=120)
        asan_errors = [
            line for line in output.splitlines()
            if "ERROR: AddressSanitizer" in line
        ]
        if asan_errors:
            logger.warning(
                "ASAN found %d issue(s) in hipEventQuery path:\n  %s",
                len(asan_errors),
                "\n  ".join(asan_errors[:10]),
            )
            # Extract unique error types
            error_types = set()
            for line in output.splitlines():
                m = re.search(r"ERROR: AddressSanitizer: (\S+)", line)
                if m:
                    error_types.add(m.group(1))
            logger.warning("Error types found: %s", ", ".join(sorted(error_types)))
        else:
            logger.info("No ASAN errors in hipEventQuery path")

    def test_hip_multi_stream(self, test_hip_binary):
        """Concurrent streams + event polling should complete."""
        rc, output = run_asan_binary(test_hip_binary, ["multi_stream"], timeout=120)
        assert rc == 0, f"Multi-stream test failed (rc={rc}):\n{output[-500:]}"

    def test_hip_device_malloc_free(self, build_dir):
        """hipMalloc/hipFree cycle under ASAN."""
        source = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            #include <cstdio>
            int main() {
                for (int i = 0; i < 50; i++) {
                    void* p;
                    hipError_t err = hipMalloc(&p, 1 << 20);
                    if (err != hipSuccess) {
                        fprintf(stderr, "hipMalloc failed: %s\\n", hipGetErrorString(err));
                        return 1;
                    }
                    err = hipFree(p);
                    if (err != hipSuccess) {
                        fprintf(stderr, "hipFree failed: %s\\n", hipGetErrorString(err));
                        return 1;
                    }
                }
                printf("hipMalloc/hipFree x50: OK\\n");
                return 0;
            }
        """)
        output = build_dir / "device_malloc"
        compile_hip_source(source, output)
        rc, out = run_asan_binary(output)
        assert rc == 0, f"hipMalloc/hipFree failed (rc={rc}):\n{out[-500:]}"
        assert "ERROR: AddressSanitizer" not in out, \
            f"ASAN error in hipMalloc/hipFree:\n{out[-500:]}"

    def test_hip_memcpy_h2d_d2h(self, build_dir):
        """hipMemcpy host-to-device and device-to-host under ASAN."""
        source = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            #include <cstdio>
            #include <cstdlib>
            int main() {
                const int N = 4096;
                float* h = (float*)malloc(N * sizeof(float));
                for (int i = 0; i < N; i++) h[i] = (float)i;
                float* d;
                hipMalloc(&d, N * sizeof(float));
                hipMemcpy(d, h, N * sizeof(float), hipMemcpyHostToDevice);
                float* h2 = (float*)malloc(N * sizeof(float));
                hipMemcpy(h2, d, N * sizeof(float), hipMemcpyDeviceToHost);
                int errs = 0;
                for (int i = 0; i < N; i++) {
                    if (h2[i] != (float)i) errs++;
                }
                hipFree(d);
                free(h);
                free(h2);
                printf("hipMemcpy round-trip: %s (%d errors)\\n",
                       errs ? "FAIL" : "OK", errs);
                return errs ? 1 : 0;
            }
        """)
        output = build_dir / "memcpy_test"
        compile_hip_source(source, output)
        rc, out = run_asan_binary(output)
        assert rc == 0, f"hipMemcpy round-trip failed:\n{out[-500:]}"
        assert "ERROR: AddressSanitizer" not in out

    def test_hip_stream_create_destroy_cycle(self, build_dir):
        """Rapid stream create/destroy under ASAN to catch lifecycle bugs."""
        source = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            #include <cstdio>
            int main() {
                for (int i = 0; i < 100; i++) {
                    hipStream_t s;
                    hipError_t err = hipStreamCreate(&s);
                    if (err != hipSuccess) {
                        fprintf(stderr, "hipStreamCreate failed at iter %d: %s\\n",
                                i, hipGetErrorString(err));
                        return 1;
                    }
                    err = hipStreamSynchronize(s);
                    if (err != hipSuccess) {
                        fprintf(stderr, "hipStreamSynchronize failed: %s\\n",
                                hipGetErrorString(err));
                        return 1;
                    }
                    err = hipStreamDestroy(s);
                    if (err != hipSuccess) {
                        fprintf(stderr, "hipStreamDestroy failed: %s\\n",
                                hipGetErrorString(err));
                        return 1;
                    }
                }
                printf("stream create/destroy x100: OK\\n");
                return 0;
            }
        """)
        output = build_dir / "stream_lifecycle"
        compile_hip_source(source, output)
        rc, out = run_asan_binary(output)
        assert rc == 0, f"Stream lifecycle test failed:\n{out[-500:]}"
        assert "ERROR: AddressSanitizer" not in out

    def test_hip_event_create_destroy_cycle(self, build_dir):
        """Rapid event create/record/query/destroy under ASAN."""
        source = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            #include <cstdio>
            int main() {
                hipStream_t s;
                hipStreamCreate(&s);
                for (int i = 0; i < 200; i++) {
                    hipEvent_t e;
                    hipEventCreate(&e);
                    hipEventRecord(e, s);
                    hipError_t status;
                    do { status = hipEventQuery(e); } while (status == hipErrorNotReady);
                    hipEventDestroy(e);
                }
                hipStreamDestroy(s);
                printf("event create/record/query/destroy x200: OK\\n");
                return 0;
            }
        """)
        output = build_dir / "event_lifecycle"
        compile_hip_source(source, output)
        rc, out = run_asan_binary(output)
        assert rc == 0, f"Event lifecycle test failed:\n{out[-500:]}"
        assert "ERROR: AddressSanitizer" not in out

    def test_hip_event_elapsed_time(self, build_dir):
        """hipEventElapsedTime between two events under ASAN."""
        source = textwrap.dedent("""\
            #include <hip/hip_runtime.h>
            #include <cstdio>
            __global__ void busy(volatile int* p, int n) {
                int x = 0;
                for (int i = 0; i < n; i++) x += i;
                if (threadIdx.x == 0) *p = x;
            }
            int main() {
                volatile int* d;
                hipMalloc((void**)&d, sizeof(int));
                hipEvent_t start, stop;
                hipEventCreate(&start);
                hipEventCreate(&stop);
                hipEventRecord(start, 0);
                busy<<<1, 256>>>(d, 1000000);
                hipEventRecord(stop, 0);
                hipEventSynchronize(stop);
                float ms = 0;
                hipEventElapsedTime(&ms, start, stop);
                printf("kernel elapsed: %.3f ms\\n", ms);
                hipEventDestroy(start);
                hipEventDestroy(stop);
                hipFree((void*)d);
                return (ms > 0) ? 0 : 1;
            }
        """)
        output = build_dir / "elapsed_time"
        compile_hip_source(source, output)
        rc, out = run_asan_binary(output)
        assert rc == 0, f"ElapsedTime test failed:\n{out[-500:]}"
        assert "ERROR: AddressSanitizer" not in out
