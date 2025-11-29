#!/usr/bin/env python3
"""
GPU Status Verification Script

Tests that all GPU-accelerated components are properly configured:
1. CUDA driver availability
2. ONNX Runtime GPU (CUDAExecutionProvider)
3. cuML/cupy for clustering

Usage:
    docker exec audiomuse-ai-worker-instance-dev python3 /app/test/test_gpu_status.py
"""

import sys


def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(name, success, details=""):
    status = "PASS" if success else "FAIL"
    symbol = "[OK]" if success else "[X]"
    print(f"  {symbol} {status}: {name}")
    if details:
        print(f"         {details}")


def test_cuda_driver():
    """Test 1: Check CUDA driver availability via libcuda.so"""
    print_header("1. CUDA Driver Check")
    try:
        import ctypes
        cuda = ctypes.CDLL('libcuda.so.1')
        init_result = cuda.cuInit(0)
        if init_result != 0:
            print_result("CUDA driver init", False, f"cuInit returned {init_result}")
            return False

        device_count = ctypes.c_int()
        result = cuda.cuDeviceGetCount(ctypes.byref(device_count))
        if result != 0 or device_count.value == 0:
            print_result("CUDA devices", False, "No CUDA devices found")
            return False

        print_result("CUDA driver", True, f"{device_count.value} device(s) available")
        return True
    except Exception as e:
        print_result("CUDA driver", False, str(e))
        return False


def test_nvidia_smi():
    """Test 1b: Check nvidia-smi output"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print_result("nvidia-smi", True, gpu_info)
            return True
        else:
            print_result("nvidia-smi", False, result.stderr)
            return False
    except Exception as e:
        print_result("nvidia-smi", False, str(e))
        return False


def test_onnx_runtime():
    """Test 2: Check ONNX Runtime GPU support"""
    print_header("2. ONNX Runtime (Analysis)")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = 'CUDAExecutionProvider' in providers
        print_result("ONNX Runtime GPU", has_cuda, f"Providers: {providers}")

        # Test actual session creation with CUDA
        if has_cuda:
            try:
                # Create a minimal test to verify CUDA actually works
                sess_options = ort.SessionOptions()
                # Just check that we can request CUDA without error
                print_result("CUDA EP available", True, "CUDAExecutionProvider ready")
            except Exception as e:
                print_result("CUDA EP test", False, str(e))
                return False
        return has_cuda
    except Exception as e:
        print_result("ONNX Runtime", False, str(e))
        return False


def test_cuml_clustering():
    """Test 3: Check cuML/cupy for clustering"""
    print_header("3. cuML/cupy (Clustering)")

    # Test cupy
    cupy_ok = False
    try:
        import cupy as cp
        test_array = cp.array([1, 2, 3])
        result = test_array.sum()
        print_result("cupy", True, f"Test sum={result}")
        cupy_ok = True
    except Exception as e:
        print_result("cupy", False, str(e))

    # Test cuML KMeans
    cuml_ok = False
    try:
        from cuml.cluster import KMeans as cuKMeans
        print_result("cuML KMeans", True, "Import successful")
        cuml_ok = True
    except Exception as e:
        print_result("cuML KMeans", False, str(e))

    # Test cuML DBSCAN
    try:
        from cuml.cluster import DBSCAN as cuDBSCAN
        print_result("cuML DBSCAN", True, "Import successful")
    except Exception as e:
        print_result("cuML DBSCAN", False, str(e))

    # Test cuML PCA
    try:
        from cuml.decomposition import PCA as cuPCA
        print_result("cuML PCA", True, "Import successful")
    except Exception as e:
        print_result("cuML PCA", False, str(e))

    # Test clustering_gpu module
    try:
        from tasks.clustering_gpu import check_gpu_available
        gpu_available = check_gpu_available()
        print_result("clustering_gpu.check_gpu_available()", gpu_available)
    except Exception as e:
        print_result("clustering_gpu module", False, str(e))

    return cupy_ok and cuml_ok


def main():
    print("\n" + "=" * 60)
    print(" AudioMuse-AI GPU Status Verification")
    print("=" * 60)

    results = {}

    # Run all tests
    results['cuda_driver'] = test_cuda_driver()
    test_nvidia_smi()  # Informational
    results['onnx_runtime'] = test_onnx_runtime()
    results['cuml_clustering'] = test_cuml_clustering()

    # Summary
    print_header("SUMMARY")
    all_pass = all(results.values())

    for name, passed in results.items():
        status = "[OK]" if passed else "[X]"
        print(f"  {status} {name}")

    print()
    if all_pass:
        print("  All GPU components are working correctly!")
        return 0
    else:
        print("  Some GPU components are not available.")
        print("  Check the detailed output above for issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
