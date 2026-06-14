import os
import yaml
import subprocess
import pandas as pd
import shutil
from run_grid_ablation import MVTEC_CLASSES

def process_calibration_ablation():
    dataset = 'mvtec'
    TARGET_MODE = '0.8'
    TS = 32
    
    results_dir = f'./results_paper_calibration'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'table7_calibration_{dataset}.csv')
    
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    processed.add(f"{parts[0]}_{parts[1]}")
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,K_Shots,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,CalibTime\n")
            
    base_config_path = f'NetworkConfigs/s2ad_configs/MVTec.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # 0 means Full dataset
    k_shots_list = [1, 2, 4, 10, 50, 100, 0]
        
    for cls in MVTEC_CLASSES:
        for k_shots in k_shots_list:
            k_key = f"{cls}_{k_shots}"
            if k_key in processed:
                continue
                
            print(f"\n{'='*60}")
            print(f"[SUBPROCESS] CALIBRATION ABLATION | {cls} | K={k_shots if k_shots > 0 else 'Full'}")
            print(f"{'='*60}")
            
            temp_config_path = os.path.join(results_dir, f'temp_{cls}_K{k_shots}.yaml')
            temp_config = base_config.copy()
            temp_config['Network']['snn_mode'] = TARGET_MODE
            temp_config['Network']['timesteps'] = [TS]
            temp_config['Network']['save_anomaly_maps'] = False
            temp_config['Network']['layers'] = 'layer123'
            temp_config['Network']['combine_method'] = 'mad_weighted'
            temp_config['Network']['use_zscore'] = True
            temp_config['Network']['calib_samples'] = k_shots
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
                
            temp_save_path = os.path.join(results_dir, f'temp_results_{cls}')
            os.makedirs(temp_save_path, exist_ok=True)
            
            cmd = [
                "python", "main_s2ad.py",
                "-name", "calib_run",
                "-category", cls,
                "-config", temp_config_path,
                "-alpha", "0.01",
                "-project_save_path", temp_save_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"\n[ERROR] Subprocess failed for {cls} K={k_shots}")
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
                        calib_time = float(parts[11])
                        
                        row = f"{cls},{k_shots},{img_auc:.4f},{img_ap:.4f},{img_f1:.4f},{pix_auc:.4f},{pix_ap:.4f},{pix_f1:.4f},{pro:.4f},{mad:.4f},{calib_time:.1f}\n"
                        
                        with open(csv_path, 'a') as fc:
                            fc.write(row)
                        processed.add(k_key)
                        
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            if os.path.exists(temp_save_path):
                shutil.rmtree(temp_save_path)
                
    df = pd.read_csv(csv_path)
    t7 = df.groupby(['K_Shots']).mean(numeric_only=True).reset_index()
    t7 = t7.sort_values(by='K_Shots')
    # Move K=0 (Full) to the bottom
    is_full = t7['K_Shots'] == 0
    t7_final = pd.concat([t7[~is_full], t7[is_full]])
    
    t7_final.to_csv(os.path.join(results_dir, f'table7_calibration_summary.csv'), index=False)
    print(f"\n[SUCCESS] Generated Table 7 Calibration summaries.")

if __name__ == '__main__':
    process_calibration_ablation()
