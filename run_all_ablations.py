import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*80}")
    print(f"🚀 BẮT ĐẦU CHẠY SCRIPT: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        # Gọi subprocess chạy bằng python executable hiện tại
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ HOÀN THÀNH SCRIPT: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ LỖI KHI CHẠY SCRIPT: {script_name}")
        print(f"Mã lỗi: {e.returncode}")
        sys.exit(1)

def main():
    print("BẮT ĐẦU TIẾN TRÌNH CHẠY LIÊN TIẾP CÁC ABLATION STUDIES...")
    print("Tiến trình sẽ chạy Module Ablation, sau đó tự động chuyển sang Layer Ablation.\n")
    
    # 1. Chạy Module Ablation
    run_script("run_paper_module_optimized.py")
    
    # 2. Chạy xong thì tự động chạy tiếp Layer Ablation
    run_script("run_layer_ablation_mains2ad.py")
    
    # 3. Chạy tiếp Calibration Sample Ablation
    run_script("run_calibration_ablation_mains2ad.py")
    
    # 4. Tính toán Hardware Footprint & Energy Ablation
    run_script("run_hardware_footprint.py")
    
    print("\n🎉 HOÀN TẤT TOÀN BỘ TIẾN TRÌNH ABLATION!")

if __name__ == '__main__':
    main()
