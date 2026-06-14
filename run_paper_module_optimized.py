import os
import time
import random
import numpy as np
import pandas as pd
import torch
import gc
from main_s2ad import BackboneEncoder, build_snn_encoder, compute_normal_stats
from run_comprehensive_ablation import run_evaluation
from datasets.load_dataset_snn import load_mvtec, load_visa
import global_v as glv
import yaml
from run_grid_ablation import MVTEC_CLASSES, VISA_CLASSES

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_paper_module(dataset, classes, config_path, results_dir, target_mode):
    os.makedirs(results_dir, exist_ok=True)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['Network']
        
    config['batch_size'] = 16
    config['input_size'] = 256
    glv.network_config = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = os.path.join(results_dir, f'table4_module_{dataset}.csv')
    processed_configs = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    processed_configs.add(f"{parts[0]}_{parts[1]}")
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Config_Name,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD\n")
            
    exp_modules = [
        {'name': 'No_mAD_No_ZScore', 'combine': 'average', 'zscore': False},
        {'name': 'mAD_Only', 'combine': 'mad_weighted', 'zscore': False},
        {'name': 'ZScore_Only', 'combine': 'average', 'zscore': True},
        {'name': 'Full_S2AD_Proposed', 'combine': 'mad_weighted', 'zscore': True}
    ]
            
    for cls in classes:
        print(f"\n[{dataset.upper()}] Class: {cls}")
        all_done = all(f"{cls}_{exp['name']}" in processed_configs for exp in exp_modules)
        if all_done: continue
        
        glv.network_config['batch_size'] = 16
        if dataset == 'mvtec':
            train_loader, test_loader = load_mvtec(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
        else:
            train_loader, test_loader = load_visa(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
            
        ann_encoder = BackboneEncoder(backbone='vgg16', layers='layer123').to(device)
        snn_encoder = build_snn_encoder(ann_encoder, train_loader, device, mode=target_mode)
        
        # We use T=32 for VGG16 as standard baseline for Table 4
        TS = 32
        normal_stats = compute_normal_stats(snn_encoder, train_loader, device, TS, 'layer123')
        
        for exp in exp_modules:
            k = f"{cls}_{exp['name']}"
            if k in processed_configs: continue
            
            m, _, _, _ = run_evaluation(
                snn_encoder, test_loader, normal_stats, device, 'layer123',
                exp['combine'], exp['zscore']
            )
            
            with open(csv_path, 'a') as f:
                f.write(f"{cls},{exp['name']},{m['img_auc']:.4f},{m['img_ap']:.4f},{m['img_f1']:.4f},{m['pix_auc']:.4f},{m['pix_ap']:.4f},{m['pix_f1']:.4f},{m['pro']:.4f},{m['mad']:.4f}\n")
            processed_configs.add(k)
            
        del normal_stats; del snn_encoder; del ann_encoder; del train_loader; del test_loader
        torch.cuda.empty_cache(); gc.collect()
        
    df = pd.read_csv(csv_path)
    t4 = df.groupby('Config_Name').mean(numeric_only=True).reset_index()
    t4.to_csv(os.path.join(results_dir, f'table4_module_{dataset}_summary.csv'), index=False)

if __name__ == '__main__':
    seed_everything(42)
    process_paper_module('mvtec', MVTEC_CLASSES, 'NetworkConfigs/s2ad_configs/MVTec.yaml', './results_paper_module_mvtec', '0.8')
    process_paper_module('visa', VISA_CLASSES, 'NetworkConfigs/s2ad_configs/VisA.yaml', './results_paper_module_visa', '0.6')
    print("\n[INFO] MODULE ABLATION FOR TABLE 4 FINISHED!")
