# ==============================================================================
# Bernasconi-Krylov Colab Benchmark Environment Setup & Build Script
# Target GPU: NVIDIA A100-SXM4-40GB / A100-80GB / H100
#
# Usage in Google Colab:
#   1. Upload colab.zip to the /content/ folder using the file browser.
#   2. Paste and run this cell. It will install dependencies, configure CMake,
#      and build the high-performance 'run_benchmark' binary.
#   3. Run your desired benchmark command in a new cell, e.g.:
#        !./build/run_benchmark configs/probability_scan_n28.cfg
# ==============================================================================
import os, sys, glob, shutil

# 1. Verify GPU & Install Dependencies
print("=== Step 1: Checking GPU & Installing Dependencies ===")
!nvidia-smi
!pip install -q cuquantum-cu12
!apt-get install -y -qq libeigen3-dev

# 2. Locate cuquantum libraries and headers
print("\n=== Step 2: Locating cuQuantum SDK ===")
so_candidates = glob.glob("/usr/local/lib/python*/dist-packages/cuquantum/lib/libcustatevec.so*")
if not so_candidates:
    so_candidates = glob.glob("/usr/local/lib/python*/dist-packages/**/libcustatevec.so*", recursive=True)
if not so_candidates:
    raise FileNotFoundError("Could not find libcustatevec.so. Please ensure 'pip install cuquantum-cu12' succeeded.")

cuquantum_so = sorted(so_candidates, key=lambda x: len(x))[0]
cuquantum_lib_dir = os.path.dirname(cuquantum_so)

inc_candidates = glob.glob("/usr/local/lib/python*/dist-packages/cuquantum/include")
if not inc_candidates:
    inc_candidates = glob.glob("/usr/local/lib/python*/dist-packages/**/custatevec.h", recursive=True)
    cuquantum_include = os.path.dirname(inc_candidates[0]) if inc_candidates else None
else:
    cuquantum_include = inc_candidates[0]

cuquantum_root = os.path.dirname(cuquantum_lib_dir)

os.environ["CUQUANTUM_ROOT"] = cuquantum_root
os.environ["LD_LIBRARY_PATH"] = f"{cuquantum_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

print(f"[+] cuQuantum Root:    {cuquantum_root}")
print(f"[+] cuStateVec Lib:    {cuquantum_so}")
print(f"[+] cuQuantum Include: {cuquantum_include}")
print(f"[+] LD_LIBRARY_PATH:   {os.environ['LD_LIBRARY_PATH']}")

# 3. Unpack project archive
print("\n=== Step 3: Unpacking Project Archive ===")
archive_candidates = ["/content/colab.zip", "/colab.zip", "colab.zip"]
archive = next((a for a in archive_candidates if os.path.exists(a)), None)

if archive:
    print(f"[+] Extracting {archive} to /content/...")
    !unzip -o -q {archive} -d /content/
else:
    print("[!] Warning: colab.zip not found in /content/ or /. Please upload colab.zip to the Colab files pane.")

# Ensure we are inside /content
%cd /content/

# 4. Build run_benchmark with CMake (Release mode)
print("\n=== Step 4: Building run_benchmark with CMake ===")
!cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCUQUANTUM_ROOT="{cuquantum_root}" \
    -DCUQUANTUM_INCLUDE_DIR="{cuquantum_include}" \
    -DCUSTATEVEC_LIB="{cuquantum_so}"

!cmake --build build --target run_benchmark -j$(nproc)

print("\n==============================================================================")
print("[✓] Environment Setup & Build Complete!")
print("    Binary ready at: /content/build/run_benchmark")
print("    Example execution commands:")
print("      !./build/run_benchmark configs/probability_scan_n28.cfg")
print("      !./build/run_benchmark configs/n26.ini")
print("==============================================================================")
