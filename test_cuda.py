import torch
import time

# --- 1. 环境设置与参数定义 ---
if not torch.cuda.is_available():
    print("CUDA is not available. This script requires a CUDA-enabled GPU.")
    exit()

device = torch.device("cuda")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# 定义矩阵大小
matrix_size = 4096 
# CPU 计算的负载
cpu_workload = 10_000_000 

# 使用锁页内存以实现最高效率的异步传输
cpu_tensor_pinned = torch.randn(matrix_size, matrix_size, device='cpu').pin_memory()

# 在 GPU 上预先分配好计算所需的张量
gpu_tensor1 = torch.randn(matrix_size, matrix_size, device=device)
gpu_tensor2 = torch.randn(matrix_size, matrix_size, device=device)
gpu_tensor_for_copy = torch.empty(matrix_size, matrix_size, device=device)


# --- 2. 定义带有详细计时的测试函数 ---

def detailed_serial_execution():
    """
    串行执行，并分别测量每个部分的耗时
    """
    # 创建 CUDA Events 用于计时
    start_event = torch.cuda.Event(enable_timing=True)
    compute_end_event = torch.cuda.Event(enable_timing=True)
    copy_end_event = torch.cuda.Event(enable_timing=True)

    # --- 任务 A: 大矩阵计算 ---
    start_event.record() # 记录开始时间点
    for _ in range(100):
        _ = torch.matmul(gpu_tensor1, gpu_tensor2)
    compute_end_event.record() # 记录计算结束时间点
    
    # 强制同步，等待 GPU 计算完成
    torch.cuda.synchronize()
    
    # --- 任务 B: CPU 计算和 HtoD 数据传输 ---
    cpu_start_time = time.time()
    for i in range(cpu_workload):
        _ = i * i
    cpu_end_time = time.time()
    
    # HtoD 传输
    gpu_tensor_for_copy.copy_(cpu_tensor_pinned, non_blocking=True)
    copy_end_event.record() # 记录拷贝结束时间点

    # 等待所有操作完成以获取准确的事件时间
    torch.cuda.synchronize()

    compute_time = start_event.elapsed_time(compute_end_event) / 1000.0 # 转换为秒
    copy_time = compute_end_event.elapsed_time(copy_end_event) / 1000.0 # 拷贝时间
    cpu_time = cpu_end_time - cpu_start_time
    total_time = compute_time + copy_time + cpu_time

    return total_time, compute_time, cpu_time, copy_time

def detailed_parallel_execution():
    """
    并行执行，并分别测量每个部分的耗时
    """
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # 为每个Stream上的操作创建独立的 Events
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    
    # 记录总体的墙上时钟时间
    overall_start_time = time.time()

    # --- 任务 A: 在 stream1 中执行计算并计时 ---
    with torch.cuda.stream(stream1):
        compute_start.record()
        for _ in range(100):
            _ = torch.matmul(gpu_tensor1, gpu_tensor2)
        compute_end.record()

    # --- 任务 B: CPU 计算 + 在 stream2 中传输并计时 ---
    cpu_start_time = time.time()
    for i in range(cpu_workload):
        _ = i * i
    cpu_end_time = time.time()
    
    with torch.cuda.stream(stream2):
        copy_start.record()
        gpu_tensor_for_copy.copy_(cpu_tensor_pinned, non_blocking=True)
        copy_end.record()

    # 等待所有 Stream 完成
    stream1.synchronize()
    stream2.synchronize()
    
    overall_end_time = time.time()

    # 计算各个部分的时间
    total_time = overall_end_time - overall_start_time
    compute_time = compute_start.elapsed_time(compute_end) / 1000.0
    cpu_time = cpu_end_time - cpu_start_time
    copy_time = copy_start.elapsed_time(copy_end) / 1000.0

    return total_time, compute_time, cpu_time, copy_time


# --- 3. 执行测试并打印详细结果 ---
if __name__ == "__main__":
    # 预热
    print("Warming up...")
    detailed_serial_execution()
    detailed_parallel_execution()
    print("Warm-up complete.\n")

    print("--- Running Detailed Serial Execution Test ---")
    s_total, s_compute, s_cpu, s_copy = detailed_serial_execution()
    print(f"  CUDA Compute Time: {s_compute:.6f} seconds")
    print(f"  CPU Work Time:     {s_cpu:.6f} seconds")
    print(f"  Memory Copy Time:  {s_copy:.6f} seconds")
    print(f"  ------------------------------------------")
    print(f"  Calculated Total:  {s_compute + s_cpu + s_copy:.6f} seconds")


    print("\n--- Running Detailed Parallel Execution Test ---")
    p_total, p_compute, p_cpu, p_copy = detailed_parallel_execution()
    print(f"  CUDA Compute Time: {p_compute:.6f} seconds")
    print(f"  CPU Work Time:     {p_cpu:.6f} seconds")
    print(f"  Memory Copy Time:  {p_copy:.6f} seconds")
    print(f"  ------------------------------------------")
    print(f"  Actual Wall Time:  {p_total:.6f} seconds")
    

    print("\n--- Analysis ---")
    print("In the serial case, the total time is the sum of all parts.")
    print("In the parallel case, the total wall time is significantly less than the sum of the parts.")
    print("This is because the CPU work, CUDA computation, and memory copy were overlapping.")
    print(f"The total time is determined by the longest path, roughly max(compute_time, cpu_time + copy_time).")
    print(f"Overlap benefit: {(s_compute + s_cpu + s_copy - p_total):.4f} seconds were saved by parallelization.")