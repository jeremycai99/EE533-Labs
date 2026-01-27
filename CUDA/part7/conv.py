import ctypes, numpy as np, time

lib = ctypes.cdll.LoadLibrary("./libcustom.so")

lib.gpu_convolve_u8.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8,   ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int
]
lib.gpu_convolve_u8.restype = ctypes.c_float  # returns kernel ms

lib.cpu_convolve_u8.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8,   ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int
]
lib.cpu_convolve_u8.restype = ctypes.c_float  # returns kernel ms

M = 2**12
N = 5
img = (np.random.rand(M, M) * 255).astype(np.uint8)

# example edge filter (Sobel X)
ker = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)

out = np.zeros((M, M), dtype=np.float32)

t0 = time.time()
ms_gpu = lib.gpu_convolve_u8(img.ravel(), ker.ravel(), out.ravel(), M, N)
t1 = time.time()

t2 = time.time()
# verify correctness with simple CPU convolution
ms_cpu = lib.cpu_convolve_u8(img.ravel(), ker.ravel(), out.ravel(), M, N)
t3 = time.time()

print(f"gpu_ms={ms_gpu:.3f}, python_total_s={t1-t0:.4f}")
print(f"cpu_ms={ms_cpu:.3f}, python_total_s={t3-t2:.4f}")