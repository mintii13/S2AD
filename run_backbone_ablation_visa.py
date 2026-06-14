import os
import time
import argparse
import gc
import random
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

import main_s2ad
from main_s2ad import BackboneEncoder, build_snn_encoder, compute_normal_stats, get_firing_rates, get_zscore_layer, get_interpolator
from datasets.load_dataset_snn import load_visa
import global_v as glv
from ad_eval import compute_pro_metric
from spikingjelly.activation_based import functional

# --- BACKBONE CONFIG ---
BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'alexnet', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11']
TIMESTEPS = [4, 8, 16, 32, 64]
MODE = '0.6'
ALPHA = 0.01

VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

def evaluate_fast_ablation(snn_encoder, test_loader, normal_stats, device, timesteps, layers, img_size, combine_method):
    start_time = time.time()
    
    total_forward_time = 0.0
    total_post_time = 0.0
    total_fr = 0.0
    num_imgs = 0
    
    results = {'img_scores': [], 'img_labels': [], 'pix_scores': [], 'pix_labels': [], 'gt_masks': [], 'anomaly_maps': []}
    
    for imgs, lbls, gt_paths in test_loader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        num_imgs += B
        
        # SNN Forward pass
        t0 = time.time()
        rates = get_firing_rates(snn_encoder, imgs, device, timesteps, layers)
        total_forward_time += time.time() - t0
        
        batch_fr = np.mean([r.mean().item() for r in rates.values()])
        total_fr += batch_fr * B
        
        t1 = time.time()
        
        deviations = {}
        for layer_name, rate in rates.items():
            hw_layer = get_zscore_layer(layer_name, normal_stats, device, use_zscore=True, alpha=ALPHA)
            with torch.no_grad():
                deviations[layer_name] = hw_layer(rate)
                
        target_name = list(deviations.keys())[0]
        target_res = deviations[target_name].shape[1:]
        weighted_sum, total_weight = None, 0.0
        
        for layer_name, dev in deviations.items():
            if dev.shape[1:] != target_res:
                interpolator = get_interpolator(dev.shape[1:], target_res, device)
                with torch.no_grad():
                    dev = interpolator(dev)
            
            if combine_method == 'mad_weighted':
                mad = normal_stats[layer_name]['mad']
                weight = 1.0 / (mad + 1e-8)
            else:
                weight = 1.0
                
            total_weight += weight
            if weighted_sum is None:
                weighted_sum = dev * weight
            else:
                weighted_sum += dev * weight
                
        score_spatial = weighted_sum / total_weight if combine_method == 'mad_weighted' else weighted_sum / len(deviations)
        final_interpolator = get_interpolator(score_spatial.shape[1:], (img_size, img_size), device)
        
        with torch.no_grad():
            score_maps = final_interpolator(score_spatial).cpu().numpy()
        batch_img_scores = [float(np.max(sm)) for sm in score_maps]
        
        for b in range(imgs.size(0)):
            lbl = lbls[b].item()
            gt_path = gt_paths[b]
            
            results['img_scores'].append(batch_img_scores[b])
            results['img_labels'].append(lbl)
            
            if lbl == 1 and gt_path and os.path.exists(gt_path):
                gt = cv2.resize(cv2.imread(gt_path, 0), (img_size, img_size))
                gt_bin = (gt > 127).astype(int)
                results['pix_scores'].extend(score_maps[b].flatten())
                results['pix_labels'].extend(gt_bin.flatten())
                results['gt_masks'].append(gt_bin)
                results['anomaly_maps'].append(score_maps[b])
                
        functional.reset_net(snn_encoder)
        del rates
        del imgs
        if 'deviations' in locals(): del deviations
        if 'score_spatial' in locals(): del score_spatial
        if 'score_maps' in locals(): del score_maps
        
        total_post_time += time.time() - t1

    fr_tb = total_fr / num_imgs
    realistic_test_time = total_forward_time + total_post_time
    fps = num_imgs / realistic_test_time

    img_auc = roc_auc_score(results['img_labels'], results['img_scores']) if len(set(results['img_labels'])) == 2 else 0.0
    img_ap = average_precision_score(results['img_labels'], results['img_scores']) if len(set(results['img_labels'])) == 2 else 0.0
    prec, rec, _ = precision_recall_curve(results['img_labels'], results['img_scores'])
    img_f1 = np.max(2 * (prec * rec) / (prec + rec + 1e-8)) if len(prec) > 0 else 0.0
    
    pix_auc = roc_auc_score(results['pix_labels'], results['pix_scores']) if results['pix_labels'] else 0.0
    pix_ap = average_precision_score(results['pix_labels'], results['pix_scores']) if results['pix_labels'] else 0.0
    if results['pix_labels']:
        pprec, prec_rec, _ = precision_recall_curve(results['pix_labels'], results['pix_scores'])
        pix_f1 = np.max(2 * (pprec * prec_rec) / (pprec + prec_rec + 1e-8)) if len(pprec) > 0 else 0.0
    else:
        pix_f1 = 0.0
        
    pro_score = compute_pro_metric(results['gt_masks'], results['anomaly_maps']) if results['gt_masks'] else 0.0
    mad_metric = (img_auc + img_ap + img_f1 + pix_auc + pix_ap + pix_f1 + pro_score) / 7.0
    
    final_metrics = {
        'img_auc': img_auc, 'img_ap': img_ap, 'img_f1': img_f1,
        'pix_auc': pix_auc, 'pix_ap': pix_ap, 'pix_f1': pix_f1,
        'pro': pro_score, 'mad': mad_metric
    }
        
    return final_metrics, realistic_test_time, fps, fr_tb

def summarize_results(csv_path, dataset_name, out_txt_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    avg_df = df.groupby(['Backbone', 'Timesteps']).mean(numeric_only=True).reset_index()
    avg_df = avg_df.sort_values(by=['Backbone', 'Timesteps'])
    
    with open(out_txt_path, 'w') as f:
        f.write(f"=== {dataset_name.upper()} Backbone Ablation Summary (Mode={MODE}, Alpha={ALPHA}) ===\n")
        f.write(f"{'Backbone':<16} | {'TS':>3} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8} | {'mAD':>8} | {'MAC(G)':>10} | {'SOP(G)':>10} | {'Calib':>7} | {'Test':>7} | {'FPS':>6}\n")
        f.write('-' * 180 + '\n')
        
        for _, row in avg_df.iterrows():
            f.write(f"{row['Backbone']:<16} | {int(row['Timesteps']):>3} | {row['ImgAUC']:8.4f} | {row['ImgAP']:8.4f} | {row['ImgF1']:8.4f} | {row['PixAUC']:8.4f} | {row['PixAP']:8.4f} | {row['PixF1']:8.4f} | {row['PRO']:8.4f} | {row['mAD']:8.4f} | {row['MAC(G)']:10.4f} | {row['SOP(G)']:10.4f} | {row['CalibTime(s)']:7.1f} | {row['TestTime(s)']:7.1f} | {row['FPS']:6.1f}\n")

def process_dataset(dataset, classes, config_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['Network']
        
    config['batch_size'] = 8
    config['input_size'] = 256
    glv.network_config = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = './results_backbone_ablation_visa'
    os.makedirs(base_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, 'raw_backbone_visa.csv')
    summary_path = os.path.join(base_dir, 'backbone_summary_visa.txt')
    
    processed_configs = set()
    if os.path.exists(csv_path):
        print(f"Found existing results file at {csv_path}. Resuming...")
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    # key: Class_Backbone_Timesteps
                    k = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    processed_configs.add(k)
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Backbone,Timesteps,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,MAC(G),SOP(G),CalibTime(s),TestTime(s),FPS\n")
            
    for cls in classes:
        print(f"\n{'='*80}")
        print(f"[{dataset.upper()}] Processing Class: {cls}")
        print(f"{'='*80}")
        
        train_loader, test_loader = load_visa(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
            
        for backbone in BACKBONES:
            # Check if all ts are done for this class+backbone
            all_done = True
            for ts in TIMESTEPS:
                if f"{cls}_{backbone}_{ts}" not in processed_configs:
                    all_done = False
                    break
            
            if all_done:
                print(f"  [Backbone={backbone}] Skipping, already processed.")
                continue
                
            print(f"  [Backbone={backbone}] Initializing ANN...")
            try:
                ann_encoder = BackboneEncoder(backbone=backbone, layers=config['layers']).to(device)
            except Exception as e:
                print(f"  Failed to initialize {backbone}: {e}. Skipping.")
                continue
            
            try:
                from thop import profile
                dummy = torch.randn(1, 3, config['input_size'], config['input_size']).to(device)
                macs, _ = profile(ann_encoder, inputs=(dummy,), verbose=False)
                macs_giga = macs / 1e9
            except:
                macs_giga = 0.0
                
            print(f"  [Backbone={backbone}] Converting to SNN (Mode={MODE})...")
            t_b = time.time()
            snn_encoder = build_snn_encoder(ann_encoder, train_loader, device, mode=MODE)
            build_time = time.time() - t_b
            
            for ts in TIMESTEPS:
                if f"{cls}_{backbone}_{ts}" in processed_configs:
                    continue
                    
                print(f"    [TS={ts}] Computing Normal Stats...")
                t_c = time.time()
                normal_stats = compute_normal_stats(snn_encoder, train_loader, device, ts, config['layers'])
                calib_time = build_time + (time.time() - t_c)
                
                functional.reset_net(snn_encoder)
                torch.cuda.empty_cache()
                
                print(f"    [TS={ts}] Running Evaluation...")
                m, test_time, fps, fr_tb = evaluate_fast_ablation(
                    snn_encoder, test_loader, normal_stats, device, ts, config['layers'], 
                    config['input_size'], config['combine_method']
                )
                
                sop_giga = macs_giga * fr_tb * ts
                
                with open(csv_path, 'a') as f:
                    f.write(f"{cls},{backbone},{ts},{m['img_auc']:.4f},{m['img_ap']:.4f},{m['img_f1']:.4f},{m['pix_auc']:.4f},{m['pix_ap']:.4f},{m['pix_f1']:.4f},{m['pro']:.4f},{m['mad']:.4f},{macs_giga:.8f},{sop_giga:.8f},{calib_time:.2f},{test_time:.2f},{fps:.2f}\n")
                
                processed_configs.add(f"{cls}_{backbone}_{ts}")
                print(f"    [TS={ts}] Completed! Checkpoint saved.")
                
                main_s2ad._interpolators_cache.clear()
                del normal_stats
                torch.cuda.empty_cache()
                gc.collect()
                
            del snn_encoder
            del ann_encoder
            torch.cuda.empty_cache()
            gc.collect()
            
        del train_loader
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    summarize_results(csv_path, dataset, summary_path)
    print(f"\n[INFO] Backbone ablation completed! Summary saved to: {summary_path}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    seed_everything(42)
    process_dataset('visa', VISA_CLASSES, 'NetworkConfigs/s2ad_configs/VisA.yaml', './results_backbone_ablation')
