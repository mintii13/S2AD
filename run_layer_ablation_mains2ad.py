import os
import yaml
import subprocess
import pandas as pd
import shutil
from run_grid_ablation import MVTEC_CLASSES

def process_layer_ablation():
    dataset = 'mvtec'
    TARGET_MODE = '0.6'
    TS = 32
    
    results_dir = f'./results_paper_layer_ablation'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'table5_layer_ablation_{dataset}.csv')
    
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    processed.add(f"{parts[0]}_{parts[1]}_{parts[2]}")
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Layers,Combine_Method,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD\n")
            
    base_config_path = f'NetworkConfigs/s2ad_configs/MVTec.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    exp_configs = [
        # Single Layer
        {'layers': 'layer1', 'combine': 'average'},
        {'layers': 'layer2', 'combine': 'average'},
        {'layers': 'layer3', 'combine': 'average'},
        # Average Combine
        {'layers': 'layer12', 'combine': 'average'},
        {'layers': 'layer13', 'combine': 'average'},
        {'layers': 'layer23', 'combine': 'average'},
        {'layers': 'layer123', 'combine': 'average'},
        # MAD Weighting (Full S2AD) - Đã chạy rồi nên bỏ qua
        # {'layers': 'layer123', 'combine': 'mad_weighted'},
    ]
        
    for cls in MVTEC_CLASSES:
        for exp in exp_configs:
            layers = exp['layers']
            combine = exp['combine']
            k = f"{cls}_{layers}_{combine}"
            if k in processed:
                continue
                
            print(f"\n{'='*60}")
            print(f"[SUBPROCESS] LAYER ABLATION | {cls} | {layers} | {combine}")
            print(f"{'='*60}")
            
            temp_config_path = os.path.join(results_dir, f'temp_{cls}_{layers}_{combine}.yaml')
            temp_config = base_config.copy()
            temp_config['Network']['snn_mode'] = TARGET_MODE
            temp_config['Network']['timesteps'] = [TS]
            temp_config['Network']['batch_size'] = 8
            temp_config['Network']['calib_samples'] = -1
            temp_config['Network']['save_anomaly_maps'] = False
            temp_config['Network']['layers'] = layers
            temp_config['Network']['combine_method'] = combine
            temp_config['Network']['use_zscore'] = True
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
                
            temp_save_path = os.path.join(results_dir, f'temp_results_{cls}')
            os.makedirs(temp_save_path, exist_ok=True)
            
            cmd = [
                "python", "main_s2ad.py",
                "-name", "layer_run",
                "-category", cls,
                "-config", temp_config_path,
                "-alpha", "0.01",
                "-seed", "42",
                "-project_save_path", temp_save_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"\n[ERROR] Subprocess failed for {cls} {layers} {combine}")
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
                        
                        row = f"{cls},{layers},{combine},{img_auc:.4f},{img_ap:.4f},{img_f1:.4f},{pix_auc:.4f},{pix_ap:.4f},{pix_f1:.4f},{pro:.4f},{mad:.4f}\n"
                        
                        with open(csv_path, 'a') as fc:
                            fc.write(row)
                        processed.add(k)
                        
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            if os.path.exists(temp_save_path):
                shutil.rmtree(temp_save_path)
                
    df = pd.read_csv(csv_path)
    t5 = df.groupby(['Layers', 'Combine_Method']).mean(numeric_only=True).reset_index()
    
    # Sort for output display matching the paper table conceptually
    def sort_key(row):
        layers = row['Layers']
        combine = row['Combine_Method']
        if combine == 'mad_weighted':
            return 3
        if layers in ['layer1', 'layer2', 'layer3']:
            return 1
        return 2
        
    t5['order'] = t5.apply(sort_key, axis=1)
    t5 = t5.sort_values(by=['order', 'Layers']).drop(columns=['order'])
    
    t5.to_csv(os.path.join(results_dir, f'table5_layer_ablation_summary.csv'), index=False)
    print(f"\n[SUCCESS] Generated Table 5 Layer Ablation summaries.")

if __name__ == '__main__':
    process_layer_ablation()
