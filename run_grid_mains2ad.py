import os
import yaml
import subprocess
import pandas as pd
import shutil

def generate_summaries(csv_path, results_dir, dataset, target_mode):
    df = pd.read_csv(csv_path)
    
    # Table 5 Summary (Target Mode across Timesteps and Alphas)
    t5 = df[df['Mode'] == float(target_mode)].groupby(['Timesteps', 'Alpha']).mean(numeric_only=True).reset_index()
    if t5.empty:
        t5 = df[df['Mode'] == target_mode].groupby(['Timesteps', 'Alpha']).mean(numeric_only=True).reset_index()
    t5.to_csv(os.path.join(results_dir, f'table5_alpha_{dataset}.csv'), index=False)
    
    # Table 6 Summary (Alpha=0.01 across Timesteps and Modes)
    t6 = df[df['Alpha'] == 0.01].groupby(['Timesteps', 'Mode']).mean(numeric_only=True).reset_index()
    t6.to_csv(os.path.join(results_dir, f'table6_mode_{dataset}.csv'), index=False)
    
    print("\n[SUCCESS] Generated Table 5 and Table 6 summaries.")

def process_dataset(dataset):
    MODES = ['max', '0.8', '0.6', '0.4', '0.2']
    TARGET_MODE = '0.6' if dataset == 'visa' else '0.8'
    TIMESTEPS = [4, 8, 16, 32, 64]
    
    results_dir = f'./results_grid_mains2ad_{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'paper_grid_{dataset}_mains2ad.csv')
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("Class,Mode,Timesteps,Alpha,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,MAC(G),SOP(G),CalibTime(s),TestTime(s),FPS\n")
            
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    processed.add(f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}")
                    
    from run_grid_ablation import MVTEC_CLASSES, VISA_CLASSES
    
    classes = MVTEC_CLASSES if dataset == 'mvtec' else VISA_CLASSES
    base_config_path = f'NetworkConfigs/s2ad_configs/{"MVTec.yaml" if dataset == "mvtec" else "VisA.yaml"}'
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    for cls in classes:
        for mode in MODES:
            alphas = [0.0, 0.01, 0.05, 0.1] if mode == TARGET_MODE else [0.01]
            
            for alpha in alphas:
                all_done = all(f"{cls}_{mode}_{t}_{alpha}" in processed for t in TIMESTEPS)
                if all_done:
                    continue
                    
                print(f"\n{'='*60}")
                print(f"[SUBPROCESS] Calling main_s2ad.py | Dataset: {dataset.upper()} | Class: {cls} | Mode: {mode} | Alpha: {alpha}")
                print(f"{'='*60}")
                
                temp_config_path = os.path.join(results_dir, f'temp_{cls}_{mode}.yaml')
                temp_config = base_config.copy()
                temp_config['Network']['snn_mode'] = mode
                temp_config['Network']['timesteps'] = TIMESTEPS
                temp_config['Network']['save_anomaly_maps'] = False
                with open(temp_config_path, 'w') as f:
                    yaml.dump(temp_config, f)
                    
                temp_save_path = os.path.join(results_dir, f'temp_results_{cls}')
                os.makedirs(temp_save_path, exist_ok=True)
                
                cmd = [
                    "python", "main_s2ad.py",
                    "-name", "grid_run",
                    "-category", cls,
                    "-config", temp_config_path,
                    "-alpha", str(alpha),
                    "-project_save_path", temp_save_path
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError:
                    print(f"\n[ERROR] Subprocess failed for {cls} Mode={mode} Alpha={alpha}")
                    return
                    
                for t in TIMESTEPS:
                    res_file = os.path.join(temp_save_path, f"{dataset}_{cls}_T{t}_ad_eval_results.txt")
                    if os.path.exists(res_file):
                        with open(res_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 3:
                                last_line = lines[-1].strip()
                                parts = [x.strip() for x in last_line.split('|')]
                                img_auc = float(parts[1])
                                img_ap = float(parts[2])
                                img_f1 = float(parts[3])
                                pix_auc = float(parts[4])
                                pix_ap = float(parts[5])
                                pix_f1 = float(parts[6])
                                pro = float(parts[7])
                                mad = float(parts[8])
                                train_s = float(parts[11])
                                test_s = float(parts[12])
                                fps = float(parts[13])
                                
                                row = f"{cls},{mode},{t},{alpha},{img_auc:.4f},{img_ap:.4f},{img_f1:.4f},{pix_auc:.4f},{pix_ap:.4f},{pix_f1:.4f},{pro:.4f},{mad:.4f},0.0,0.0,{train_s:.1f},{test_s:.2f},{fps:.2f}\n"
                                
                                with open(csv_path, 'a') as fc:
                                    fc.write(row)
                                processed.add(f"{cls}_{mode}_{t}_{alpha}")
                                
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                if os.path.exists(temp_save_path):
                    shutil.rmtree(temp_save_path)
                    
    generate_summaries(csv_path, results_dir, dataset, TARGET_MODE)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['mvtec', 'visa', 'all'])
    args = parser.parse_args()
    
    if args.dataset == 'all':
        print("\n[INFO] Running Grid Ablation for BOTH MVTec and VisA...\n")
        process_dataset('mvtec')
        process_dataset('visa')
    else:
        process_dataset(args.dataset)

if __name__ == '__main__':
    main()
