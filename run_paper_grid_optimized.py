import os
import time
import random
import numpy as np
import pandas as pd
import torch
import gc
from main_s2ad import BackboneEncoder, build_snn_encoder, compute_normal_stats
from run_grid_ablation import evaluate_fast_ablation, MODES, ALPHAS, TIMESTEPS, MVTEC_CLASSES, VISA_CLASSES
from datasets.load_dataset_snn import load_mvtec, load_visa
import global_v as glv
import yaml

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_paper_grid(dataset, classes, config_path, results_dir, target_mode):
    os.makedirs(results_dir, exist_ok=True)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['Network']
        
    config['batch_size'] = 16
    config['input_size'] = 256
    glv.network_config = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = os.path.join(results_dir, f'paper_grid_{dataset}.csv')
    processed_configs = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    processed_configs.add(f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}")
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Mode,Timesteps,Alpha,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,MAC(G),SOP(G),CalibTime(s),TestTime(s),FPS\n")
            
    for cls in classes:
        print(f"\n[{dataset.upper()}] Class: {cls}")
        glv.network_config['batch_size'] = 16
        if dataset == 'mvtec':
            train_loader, test_loader = load_mvtec(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
        else:
            train_loader, test_loader = load_visa(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
            
        ann_encoder = BackboneEncoder(backbone='vgg16', layers='layer123').to(device)
        try:
            from thop import profile
            macs, _ = profile(ann_encoder, inputs=(torch.randn(1, 3, 256, 256).to(device),), verbose=False)
            macs_giga = macs / 1e9
        except: macs_giga = 0.0

        for mode in ['0.1', '0.2', '0.4', '0.6', '0.8', '0.9', '0.99', '1.0']:
            # We ONLY need Table 5 (Target Mode + Alphas) OR Table 6 (Alpha=0.01 + All Modes)
            is_target_mode = (mode == target_mode)
            needed_alphas = [0.0, 0.01, 0.05, 0.1] if is_target_mode else [0.01]
            
            # Check if done
            all_done = True
            for ts in [4, 8, 16, 32, 64]:
                for a in needed_alphas:
                    if f"{cls}_{mode}_{ts}_{a}" not in processed_configs:
                        all_done = False; break
            if all_done: continue
            
            # RE-INITIALIZE ANN ENCODER PER MODE TO PREVENT HOOK CORRUPTION
            ann_encoder = BackboneEncoder(backbone='vgg16', layers='layer123').to(device)
            snn_encoder = build_snn_encoder(ann_encoder, train_loader, device, mode=mode)
            for ts in [4, 8, 16, 32, 64]:
                ts_done = True
                for a in needed_alphas:
                    if f"{cls}_{mode}_{ts}_{a}" not in processed_configs:
                        ts_done = False; break
                if ts_done: continue
                
                normal_stats = compute_normal_stats(snn_encoder, train_loader, device, ts, 'layer123')
                m_alpha, test_time, fps, fr_tb = evaluate_fast_ablation(snn_encoder, test_loader, normal_stats, device, ts, 'layer123', 256, 'mad_weighted', needed_alphas)
                sop_giga = macs_giga * fr_tb * ts
                
                with open(csv_path, 'a') as f:
                    for alpha in needed_alphas:
                        m = m_alpha[alpha]
                        f.write(f"{cls},{mode},{ts},{alpha},{m['img_auc']:.4f},{m['img_ap']:.4f},{m['img_f1']:.4f},{m['pix_auc']:.4f},{m['pix_ap']:.4f},{m['pix_f1']:.4f},{m['pro']:.4f},{m['mad']:.4f},{macs_giga:.8f},{sop_giga:.8f},0.0,{test_time:.2f},{fps:.2f}\n")
                        processed_configs.add(f"{cls}_{mode}_{ts}_{alpha}")
                del normal_stats; torch.cuda.empty_cache(); gc.collect()
            del snn_encoder; torch.cuda.empty_cache(); gc.collect()
        del ann_encoder; del train_loader; del test_loader; torch.cuda.empty_cache(); gc.collect()
    
    # Generate Summaries
    df = pd.read_csv(csv_path)
    # Table 5 Summary
    t5 = df[df['Mode'] == float(target_mode)].groupby(['Timesteps', 'Alpha']).mean(numeric_only=True).reset_index()
    t5.to_csv(os.path.join(results_dir, f'table5_alpha_{dataset}.csv'), index=False)
    # Table 6 Summary
    t6 = df[df['Alpha'] == 0.01].groupby(['Timesteps', 'Mode']).mean(numeric_only=True).reset_index()
    t6.to_csv(os.path.join(results_dir, f'table6_mode_{dataset}.csv'), index=False)

if __name__ == '__main__':
    seed_everything(42)
    process_paper_grid('mvtec', MVTEC_CLASSES, 'NetworkConfigs/s2ad_configs/MVTec.yaml', './results_paper_grid_mvtec', '0.8')
    process_paper_grid('visa', VISA_CLASSES, 'NetworkConfigs/s2ad_configs/VisA.yaml', './results_paper_grid_visa', '0.6')
    print("\n[INFO] ALL OPTIMIZED GRID ABLATIONS FINISHED! Summaries saved to table5 and table6 csvs.")
