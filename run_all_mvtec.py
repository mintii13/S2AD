import os
import yaml
import subprocess
import argparse

CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

TIMESTEPS = [16, 8, 4]
CONFIG_PATH = 'NetworkConfigs/esvae_configs/MVTec.yaml'
RESULTS_DIR = './results_esvae'
SUMMARY_FILE = os.path.join(RESULTS_DIR, 'mvtec_overall_summary.txt')

def update_yaml_nsteps(path, n_steps):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    data['Network']['n_steps'] = n_steps
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def get_last_mad(dataset_name, category, n_steps):
    # Example: mvtec_bottle_T16_ad_eval_results.txt
    file_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{category}_T{n_steps}_ad_eval_results.txt")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('-') and not line.startswith('Epoch')]
            if not lines:
                return None
            last_line = lines[-1]
            parts = last_line.split('|')
            if len(parts) >= 9:
                mad = float(parts[-1].strip())
                return mad
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

import glob

def get_checkpoint_path(cls, config_path, t):
    # We include T{t} in the glob pattern to make sure we don't accidentally resume T=8 from a T=16 checkpoint
    pattern = os.path.join(RESULTS_DIR, 'checkpoint', 'mvtec', f'mvtec_{cls}_esvae_T{t}_*', 'checkpoint.pth')
    matches = glob.glob(pattern)
    if not matches:
        # Fallback to vanilla prefix if needed
        pattern = os.path.join(RESULTS_DIR, 'checkpoint', 'mvtec', f'vanilla_mvtec_esvae_*', 'checkpoint.pth')
        matches = glob.glob(pattern)
        
    if matches:
        # If there are multiple, sort by modification time and get the latest
        matches.sort(key=os.path.getmtime)
        return matches[-1]
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume', action='store_true', help='Auto find and resume from checkpoint')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write("=== MVTec Overall Summary ===\n")
        
    for t in TIMESTEPS:
        print(f"\n{'='*50}")
        print(f" STARTING ALL CLASSES WITH TIMESTEP T={t}")
        print(f"{'='*50}")
        update_yaml_nsteps(CONFIG_PATH, t)
        
        mADs = {}
        
        for cls in CLASSES:
            print(f"\n---> Training class: [{cls}] with T={t}")
            cmd = [
                "python", "main_esvae.py",
                "-name", "exp_name",
                "-category", cls,
                "-config", CONFIG_PATH,
                "-project_save_path", RESULTS_DIR
            ]
            
            if args.resume:
                ckpt_path = get_checkpoint_path(cls, CONFIG_PATH, t)
                if os.path.exists(ckpt_path):
                    print(f"   [RESUME] Found checkpoint: {ckpt_path}")
                    cmd.extend(["-checkpoint", ckpt_path])
            
            subprocess.run(cmd)
            
            # Đọc mAD cuối cùng sau khi train xong class này
            mad = get_last_mad('mvtec', cls, t)
            if mad is not None:
                mADs[cls] = mad
                print(f"[OK] Class '{cls}' finished! mAD: {mad:.4f}")
            else:
                print(f"[WARNING] Class '{cls}' finished but could not read mAD!")
                
        # Tính Average mAD cho Timestep này
        valid_mads = [v for v in mADs.values() if v is not None]
        avg_mad = sum(valid_mads) / len(valid_mads) if valid_mads else 0.0
        
        # Ghi vào file tổng kết
        with open(SUMMARY_FILE, 'a') as f:
            f.write(f"\nTimestep T={t}\n")
            f.write("-" * 30 + "\n")
            for cls in CLASSES:
                val = mADs.get(cls, 'N/A')
                val_str = f"{val:.4f}" if isinstance(val, float) else val
                f.write(f"{cls:<15}: {val_str}\n")
            f.write("-" * 30 + "\n")
            f.write(f"AVERAGE mAD    : {avg_mad:.4f}\n")
            f.write("=" * 30 + "\n")
            
        print(f"\n[TIMESTEP {t} COMPLETED] Average mAD across {len(valid_mads)} classes: {avg_mad:.4f}\n")

if __name__ == '__main__':
    main()
