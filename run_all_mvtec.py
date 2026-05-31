import os
import yaml
import subprocess
import argparse

CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

TIMESTEPS = [16]
CONFIG_PATH = 'NetworkConfigs/esvae_configs/MVTec.yaml'
RESULTS_DIR = './results_esvae_d1024'

def update_yaml_nsteps(path, n_steps):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    data['Network']['n_steps'] = n_steps
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def get_last_metrics(dataset_name, category, n_steps, results_dir):
    # Example: mvtec_bottle_T16_ad_eval_results.txt
    file_path = os.path.join(results_dir, f"{dataset_name}_{category}_T{n_steps}_ad_eval_results.txt")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('-') and not line.startswith('Epoch')]
            if not lines:
                return None
            last_line = lines[-1]
            parts = [p.strip() for p in last_line.split('|')]
            if len(parts) >= 14:
                return {
                    'img_auc': float(parts[1]),
                    'img_ap': float(parts[2]),
                    'img_f1': float(parts[3]),
                    'pix_auc': float(parts[4]),
                    'pix_ap': float(parts[5]),
                    'pix_f1': float(parts[6]),
                    'pro': float(parts[7]),
                    'mad': float(parts[8]),
                    'train_loss': float(parts[9]),
                    'test_loss': float(parts[10]),
                    'train_time': float(parts[11]),
                    'test_time': float(parts[12]),
                    'fps': float(parts[13])
                }
            elif len(parts) >= 11:
                return {
                    'img_auc': float(parts[1]),
                    'img_ap': float(parts[2]),
                    'img_f1': float(parts[3]),
                    'pix_auc': float(parts[4]),
                    'pix_ap': float(parts[5]),
                    'pix_f1': float(parts[6]),
                    'pro': float(parts[7]),
                    'mad': float(parts[8]),
                    'train_loss': 0.0,
                    'test_loss': 0.0,
                    'train_time': 0.0,
                    'test_time': float(parts[9]),
                    'fps': float(parts[10])
                }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

import glob

def get_checkpoint_path(cls, config_path, t, model_name, results_dir):
    # We include T{t} in the glob pattern to make sure we don't accidentally resume T=8 from a T=16 checkpoint
    pattern = os.path.join(results_dir, 'checkpoint', 'mvtec', f'mvtec_{cls}_{model_name}_T{t}_*', 'checkpoint.pth')
    matches = glob.glob(pattern)
    if not matches:
        # Fallback to vanilla prefix if needed
        pattern = os.path.join(results_dir, 'checkpoint', 'mvtec', f'vanilla_mvtec_{model_name}_*', 'checkpoint.pth')
        matches = glob.glob(pattern)
        
    if matches:
        # If there are multiple, sort by modification time and get the latest
        matches.sort(key=os.path.getmtime)
        return matches[-1]
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume', action='store_true', help='Auto find and resume from checkpoint')
    parser.add_argument('-model', type=str, default='esvae', choices=['esvae', 'fsvae'], help='Model to train (esvae or fsvae)')
    args = parser.parse_args()

    if args.model == 'esvae':
        config_path = 'NetworkConfigs/esvae_configs/MVTec.yaml'
        results_dir = './results_esvae_d1024'
        main_script = 'main_esvae.py'
    else:
        config_path = 'NetworkConfigs/fsvae_configs/MVTec.yaml'
        results_dir = './results_fsvae_d1024'
        main_script = 'main_fsvae.py'

    os.makedirs(results_dir, exist_ok=True)
    
    import time
    
    for t in TIMESTEPS:
        summary_file = os.path.join(results_dir, f'mvtec_overall_summary_T{t}.txt')
        with open(summary_file, 'w') as f:
            f.write(f"=== MVTec Overall Summary [Timestep {t}] ===\n")
            f.write(f"{'Class':<15} | {'Img AUC':>7} | {'Img AP':>7} | {'Img F1':>7} | {'Pix AUC':>7} | {'Pix AP':>7} | {'Pix F1':>7} | {'PRO':>7} | {'mAD':>7} | {'TrainLoss':>9} | {'TestLoss':>9} | {'Train(s)':>8} | {'Test(s)':>7} | {'FPS':>7}\n")
            f.write("-" * 145 + "\n")
            
        print(f"\n{'='*50}")
        print(f" STARTING ALL CLASSES WITH TIMESTEP T={t} MODEL={args.model.upper()}")
        print(f"{'='*50}")
        update_yaml_nsteps(config_path, t)
        
        all_metrics = {}
        
        for cls in CLASSES:
            print(f"\n---> Training class: [{cls}] with T={t}")
            cmd = [
                "python", main_script,
                "-name", "exp_name",
                "-category", cls,
                "-config", config_path,
                "-project_save_path", results_dir
            ]
            
            if args.resume:
                ckpt_path = get_checkpoint_path(cls, config_path, t, args.model, results_dir)
                if os.path.exists(ckpt_path):
                    print(f"   [RESUME] Found checkpoint: {ckpt_path}")
                    cmd.extend(["-checkpoint", ckpt_path])
            
            start_train = time.time()
            subprocess.run(cmd)
            train_time = time.time() - start_train
            
            metrics = get_last_metrics('mvtec', cls, t, results_dir)
            if metrics is not None:
                # We use cumulative train_time parsed from the txt if available, otherwise fallback
                train_time_to_report = metrics['train_time'] if metrics.get('train_time', 0) > 0 else train_time
                metrics['train_time'] = train_time_to_report
                all_metrics[cls] = metrics
                print(f"[OK] Class '{cls}' finished! mAD: {metrics['mad']:.4f} | Cumulative Train Time: {train_time_to_report:.1f}s")
                
                with open(summary_file, 'a') as f:
                    m = metrics
                    f.write(f"{cls:<15} | {m['img_auc']:7.4f} | {m['img_ap']:7.4f} | {m['img_f1']:7.4f} | {m['pix_auc']:7.4f} | {m['pix_ap']:7.4f} | {m['pix_f1']:7.4f} | {m['pro']:7.4f} | {m['mad']:7.4f} | {m['train_loss']:9.4f} | {m['test_loss']:9.4f} | {m['train_time']:8.1f} | {m['test_time']:7.1f} | {m['fps']:7.1f}\n")
            else:
                print(f"[WARNING] Class '{cls}' finished but could not read metrics!")
                
        valid_mads = [m['mad'] for m in all_metrics.values()]
        avg_mad = sum(valid_mads) / len(valid_mads) if valid_mads else 0.0
        
        # Calculate averages for all columns
        avg_metrics = {}
        for key in ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_ap', 'pix_f1', 'pro', 'mad', 'train_loss', 'test_loss', 'train_time', 'test_time', 'fps']:
            vals = [m[key] for m in all_metrics.values()]
            avg_metrics[key] = sum(vals) / len(vals) if vals else 0.0
            
        with open(summary_file, 'a') as f:
            f.write("=" * 145 + "\n")
            am = avg_metrics
            f.write(f"{'AVERAGE':<15} | {am['img_auc']:7.4f} | {am['img_ap']:7.4f} | {am['img_f1']:7.4f} | {am['pix_auc']:7.4f} | {am['pix_ap']:7.4f} | {am['pix_f1']:7.4f} | {am['pro']:7.4f} | {am['mad']:7.4f} | {am['train_loss']:9.4f} | {am['test_loss']:9.4f} | {am['train_time']:8.1f} | {am['test_time']:7.1f} | {am['fps']:7.1f}\n")
            
        print(f"\n[TIMESTEP {t} COMPLETED] Average mAD across {len(valid_mads)} classes: {avg_mad:.4f}\n")

if __name__ == '__main__':
    main()
