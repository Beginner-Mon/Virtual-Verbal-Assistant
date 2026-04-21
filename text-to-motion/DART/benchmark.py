import time
import argparse
import sys
import torch

try:
    from api_server import MotionGenerator
except ImportError:
    print("Error: Could not import MotionGenerator from api_server.py.")
    print("Please make sure you are running this script from the text-to-motion/DART directory.")
    sys.exit(1)

def run_benchmarks(num_runs=3, duration_seconds=2.0, text_prompt="walk forward"):
    print(f"Starting DART Benchmark")
    print(f"Parameters: prompt='{text_prompt}', duration={duration_seconds}s, runs={num_runs}")
    print("-" * 50)
    
    # 1. Benchmark CPU
    print("\n[CPU] Initializing MotionGenerator on CPU...")
    start_load_cpu = time.perf_counter()
    gen_cpu = MotionGenerator(device="cpu")
    # Load default models (uses same args as server fallback)
    gen_cpu.load_models()
    cpu_load_time = time.perf_counter() - start_load_cpu
    print(f"[CPU] Model Load Time: {cpu_load_time:.2f}s")
    
    cpu_gen_times = []
    print("[CPU] Running warmup...")
    # Warmup
    gen_cpu.generate(
        text_prompt=text_prompt, 
        duration_seconds=duration_seconds, 
        guidance_scale=5.0, 
        num_steps=50
    )
    
    for i in range(num_runs):
        print(f"[CPU] Run {i+1}/{num_runs}...")
        start_gen = time.perf_counter()
        gen_cpu.generate(
            text_prompt=text_prompt, 
            duration_seconds=duration_seconds, 
            guidance_scale=5.0, 
            num_steps=50
        )
        cpu_gen_times.append(time.perf_counter() - start_gen)
    
    avg_cpu = sum(cpu_gen_times) / num_runs
    print(f"[CPU] Average Gen Time: {avg_cpu:.2f}s")
    
    # Clean up memory
    del gen_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Benchmark GPU
    if torch.cuda.is_available():
        print("\n[GPU] Initializing MotionGenerator on CUDA...")
        start_load_gpu = time.perf_counter()
        gen_gpu = MotionGenerator(device="cuda")
        gen_gpu.load_models()
        gpu_load_time = time.perf_counter() - start_load_gpu
        print(f"[GPU] Model Load Time: {gpu_load_time:.2f}s")
        
        gpu_gen_times = []
        print("[GPU] Running warmup...")
        # Warmup
        gen_gpu.generate(
            text_prompt=text_prompt, 
            duration_seconds=duration_seconds, 
            guidance_scale=5.0, 
            num_steps=50
        )
        
        for i in range(num_runs):
            print(f"[GPU] Run {i+1}/{num_runs}...")
            start_gen = time.perf_counter()
            gen_gpu.generate(
                text_prompt=text_prompt, 
                duration_seconds=duration_seconds, 
                guidance_scale=5.0, 
                num_steps=50
            )
            # Make sure cuda ops are finished before stopping timer
            torch.cuda.synchronize()
            gpu_gen_times.append(time.perf_counter() - start_gen)
            
        avg_gpu = sum(gpu_gen_times) / num_runs
        print(f"[GPU] Average Gen Time: {avg_gpu:.2f}s")
    else:
        print("\n[GPU] CUDA is not available. Skipping GPU benchmark.")
        gpu_load_time = None
        avg_gpu = None
        
    print("\n" + "="*50)
    print("                   SUMMARY")
    print("="*50)
    print(f"Prompt  : '{text_prompt}'")
    print(f"Duration: {duration_seconds}s")
    print(f"Runs    : {num_runs}")
    print("-" * 50)
    print(f"{'DEVICE':<10} | {'LOAD TIME':<15} | {'AVG GEN TIME':<15}")
    print("-" * 50)
    print(f"{'CPU':<10} | {cpu_load_time:<12.2f} s | {avg_cpu:<12.2f} s")
    if torch.cuda.is_available():
        print(f"{'GPU':<10} | {gpu_load_time:<12.2f} s | {avg_gpu:<12.2f} s")
        speedup = avg_cpu / avg_gpu
        print("-" * 50)
        print(f"Speedup: GPU is {speedup:.2f}x faster at generation")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DART generation on CPU vs GPU")
    parser.add_argument("--runs", type=int, default=3, help="Number of profiling runs per device")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of motion in seconds")
    parser.add_argument("--prompt", type=str, default="walk forward", help="Text prompt to generate")
    args = parser.parse_args()
    
    run_benchmarks(num_runs=args.runs, duration_seconds=args.duration, text_prompt=args.prompt)
