import torch
import psutil

def get_cuda_memory_usage():
    """
    Retrieves the current CUDA memory usage.

    Returns:
        A dictionary containing allocated and cached memory in MB, or None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None

    allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    cached_memory = torch.cuda.memory_reserved() / (1024 * 1024) # Convert to MB

    return {
        "allocated_mb": allocated_memory,
        "cached_mb": cached_memory,
    }

def get_cpu_memory_usage():
    """
    Retrieves the current CPU memory usage.

    Returns:
        A dictionary containing virtual memory usage details in MB.
    """
    memory = psutil.virtual_memory()
    total_mb = memory.total / (1024 * 1024)
    available_mb = memory.available / (1024 * 1024)
    used_mb = memory.used / (1024 * 1024)
    percent_used = memory.percent

    return {
        "total_mb": total_mb,
        "available_mb": available_mb,
        "used_mb": used_mb,
        "percent_used": percent_used,
    }

if __name__ == "__main__":
    cuda_memory = get_cuda_memory_usage()
    cpu_memory = get_cpu_memory_usage()

    if cuda_memory:
        print("CUDA Memory Usage:")
        print(f"  Allocated: {cuda_memory['allocated_mb']:.2f} MB")
        print(f"  Cached: {cuda_memory['cached_mb']:.2f} MB")
    else:
        print("CUDA is not available.")

    print("\nCPU Memory Usage:")
    print(f"  Total: {cpu_memory['total_mb']:.2f} MB")
    print(f"  Available: {cpu_memory['available_mb']:.2f} MB")
    print(f"  Used: {cpu_memory['used_mb']:.2f} MB")
    print(f"  Percent Used: {cpu_memory['percent_used']:.2f} %")