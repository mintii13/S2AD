import os
import yaml
import subprocess
import pandas as pd
import shutil
from run_grid_ablation import MVTEC_CLASSES, VISA_CLASSES

def process_paper_module(dataset):
    TARGET_MODE = '0.6'
    TS = 32
    
    results_dir = f'./results_paper_module_{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'table4_module_{dataset}.csv')
    
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    processed.add(f"{parts[0]}_{parts[1]}")
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Config_Name,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD\n")
            
    classes = MVTEC_CLASSES if dataset == 'mvtec' else VISA_CLASSES
    base_config_path = f'NetworkConfigs/s2ad_configs/{"MVTec.yaml" if dataset == "mvtec" else "VisA.yaml"}'
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    exp_modules = [
        {'name': 'No_mAD_No_ZScore', 'combine': 'average', 'zscore': False},
        {'name': 'mAD_Only', 'combine': 'mad_weighted', 'zscore': False},
    ]
    if dataset == 'visa':
        exp_modules.append({'name': 'ZScore_Only', 'combine': 'average', 'zscore': True})
        
    for cls in classes:
        for exp in exp_modules:
            config_name = exp['name']
            k = f"{cls}_{config_name}"
            if k in processed:
                continue
                
            print(f"\n{'='*60}")
            print(f"[SUBPROCESS] main_s2ad.py | {dataset.upper()} | {cls} | {config_name}")
            print(f"{'='*60}")
            
            temp_config_path = os.path.join(results_dir, f'temp_{cls}_{config_name}.yaml')
            temp_config = base_config.copy()
            temp_config['Network']['snn_mode'] = TARGET_MODE
            temp_config['Network']['timesteps'] = [TS]
            temp_config['Network']['batch_size'] = 8
            temp_config['Network']['calib_samples'] = -1
            temp_config['Network']['save_anomaly_maps'] = False
            temp_config['Network']['combine_method'] = exp['combine']
            temp_config['Network']['use_zscore'] = exp['zscore']
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
                
            temp_save_path = os.path.join(results_dir, f'temp_results_{cls}')
            os.makedirs(temp_save_path, exist_ok=True)
            
            cmd = [
                "python", "main_s2ad.py",
                "-name", "module_run",
                "-category", cls,
                "-config", temp_config_path,
                "-alpha", "0.01",
                "-seed", "42",
                "-project_save_path", temp_save_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"\n[ERROR] Subprocess failed for {cls} Config={config_name}")
                continue
                
            res_file = os.path.join(temp_save_path, f"{dataset}_{cls}_T{TS}_ad_eval_results.txt")
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
                        
                        row = f"{cls},{config_name},{img_auc:.4f},{img_ap:.4f},{img_f1:.4f},{pix_auc:.4f},{pix_ap:.4f},{pix_f1:.4f},{pro:.4f},{mad:.4f}\n"
                        
                        with open(csv_path, 'a') as fc:
                            fc.write(row)
                        processed.add(k)
                        
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            if os.path.exists(temp_save_path):
                shutil.rmtree(temp_save_path)
                
    df = pd.read_csv(csv_path)
    t4 = df.groupby('Config_Name').mean(numeric_only=True).reset_index()
    t4.to_csv(os.path.join(results_dir, f'table4_module_{dataset}_summary.csv'), index=False)
    print(f"\n[SUCCESS] Generated Table 4 module summaries for {dataset}.")

def main():
    print("\n[INFO] Running Module Ablation for BOTH MVTec and VisA...\n")
    process_paper_module('mvtec')
    process_paper_module('visa')

if __name__ == '__main__':
    main()

