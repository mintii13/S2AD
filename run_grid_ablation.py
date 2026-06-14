import os
import time
import random
import argparse
import gc
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

import main_s2ad
from main_s2ad import BackboneEncoder, build_snn_encoder, compute_normal_stats, get_firing_rates, get_zscore_layer, get_interpolator
from datasets.load_dataset_snn import load_mvtec, load_visa
import global_v as glv
from ad_eval import compute_pro_metric
from spikingjelly.activation_based import functional

# --- GRID CONFIG ---
MODES = ['max', '0.99', '0.9', '0.8', '0.6', '0.4', '0.2', '0.1']
TIMESTEPS = [4, 8, 16, 32, 64]
ALPHAS = [0.0, 0.01, 0.05, 0.1]

MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
    'pipe_fryum'
]

def evaluate_fast_ablation(snn_encoder, test_loader, normal_stats, device, timesteps, layers, img_size, combine_method, alphas):
    start_time = time.time()
    
    total_forward_time = 0.0
    total_post_time = 0.0
    total_fr = 0.0
    num_imgs = 0
    
    results_by_alpha = {alpha: {'img_scores': [], 'img_labels': [], 'pix_scores': [], 'pix_labels': [], 'gt_masks': [], 'anomaly_maps': []} for alpha in alphas}
    
    for imgs, lbls, gt_paths in test_loader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        num_imgs += B
        
        # SNN Forward pass ONCE per batch
        t0 = time.time()
        rates = get_firing_rates(snn_encoder, imgs, device, timesteps, layers)
        total_forward_time += time.time() - t0
        
        batch_fr = np.mean([r.mean().item() for r in rates.values()])
        total_fr += batch_fr * B
        
        t1 = time.time()
        
        # Loop over alphas (Fast)
        for alpha in alphas:
            deviations = {}
            for layer_name, rate in rates.items():
                hw_layer = get_zscore_layer(layer_name, normal_stats, device, use_zscore=True, alpha=alpha)
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
            
            # Store results for PRO metric and AUC
            for b in range(imgs.size(0)):
                lbl = lbls[b].item()
                gt_path = gt_paths[b]
                
                results_by_alpha[alpha]['img_scores'].append(batch_img_scores[b])
                results_by_alpha[alpha]['img_labels'].append(lbl)
                
                if lbl == 1 and gt_path and os.path.exists(gt_path):
                    gt = cv2.resize(cv2.imread(gt_path, 0), (img_size, img_size))
                    gt_bin = (gt > 127).astype(int)
                    results_by_alpha[alpha]['pix_scores'].extend(score_maps[b].flatten())
                    results_by_alpha[alpha]['pix_labels'].extend(gt_bin.flatten())
                    results_by_alpha[alpha]['gt_masks'].append(gt_bin)
                    results_by_alpha[alpha]['anomaly_maps'].append(score_maps[b])
                    
        # === MEMORY CLEANUP PER BATCH ===
        functional.reset_net(snn_encoder)
        del rates
        del imgs
        if 'deviations' in locals(): del deviations
        if 'score_spatial' in locals(): del score_spatial
        if 'score_maps' in locals(): del score_maps
        
        total_post_time += time.time() - t1

    fr_tb = total_fr / num_imgs
    realistic_test_time = total_forward_time + (total_post_time / len(alphas))
    fps = num_imgs / realistic_test_time

    # Compute final metrics
    final_metrics = {}
    for alpha in alphas:
        r = results_by_alpha[alpha]
        img_auc = roc_auc_score(r['img_labels'], r['img_scores']) if len(set(r['img_labels'])) == 2 else 0.0
        img_ap = average_precision_score(r['img_labels'], r['img_scores']) if len(set(r['img_labels'])) == 2 else 0.0
        prec, rec, _ = precision_recall_curve(r['img_labels'], r['img_scores'])
        img_f1 = np.max(2 * (prec * rec) / (prec + rec + 1e-8)) if len(prec) > 0 else 0.0
        
        pix_auc = roc_auc_score(r['pix_labels'], r['pix_scores']) if r['pix_labels'] else 0.0
        pix_ap = average_precision_score(r['pix_labels'], r['pix_scores']) if r['pix_labels'] else 0.0
        if r['pix_labels']:
            pprec, prec_rec, _ = precision_recall_curve(r['pix_labels'], r['pix_scores'])
            pix_f1 = np.max(2 * (pprec * prec_rec) / (pprec + prec_rec + 1e-8)) if len(pprec) > 0 else 0.0
        else:
            pix_f1 = 0.0
            
        pro_score = compute_pro_metric(r['gt_masks'], r['anomaly_maps']) if r['gt_masks'] else 0.0
        mad_metric = (img_auc + img_ap + img_f1 + pix_auc + pix_ap + pix_f1 + pro_score) / 7.0
        
        final_metrics[alpha] = {
            'img_auc': img_auc, 'img_ap': img_ap, 'img_f1': img_f1,
            'pix_auc': pix_auc, 'pix_ap': pix_ap, 'pix_f1': pix_f1,
            'pro': pro_score, 'mad': mad_metric
        }
        
    return final_metrics, realistic_test_time, fps, fr_tb

def summarize_results(csv_path, dataset_name, out_txt_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    # Calculate mean across classes
    avg_df = df.groupby(['Mode', 'Timesteps', 'Alpha']).mean(numeric_only=True).reset_index()
    
    def mode_sort_key(m):
        if m == 'max': return 1.0
        return float(m)
    
    avg_df['mode_key'] = avg_df['Mode'].apply(mode_sort_key)
    avg_df = avg_df.sort_values(by=['mode_key', 'Timesteps', 'Alpha'], ascending=[False, True, True])
    
    # Add default columns if missing
    for col in ['MAC(G)', 'SOP(G)', 'CalibTime(s)', 'TestTime(s)', 'FPS']:
        if col not in avg_df.columns:
            avg_df[col] = 0.0
            
    with open(out_txt_path, 'w') as f:
        f.write(f"=== {dataset_name.upper()} Average Metrics across all classes ===\n")
        f.write(f"{'Mode':>5} | {'TS':>3} | {'Alpha':>5} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8} | {'mAD':>8} | {'MAC(G)':>12} | {'SOP(G)':>12} | {'Calib':>7} | {'Test':>7} | {'FPS':>6}\n")
        f.write('-' * 190 + '\n')
        
        for _, row in avg_df.iterrows():
            f.write(f"{row['Mode']:>5} | {int(row['Timesteps']):>3} | {row['Alpha']:>5.2f} | {row['ImgAUC']:8.4f} | {row['ImgAP']:8.4f} | {row['ImgF1']:8.4f} | {row['PixAUC']:8.4f} | {row['PixAP']:8.4f} | {row['PixF1']:8.4f} | {row['PRO']:8.4f} | {row['mAD']:8.4f} | {row['MAC(G)']:12.8f} | {row['SOP(G)']:12.8f} | {row['CalibTime(s)']:7.1f} | {row['TestTime(s)']:7.1f} | {row['FPS']:6.1f}\n")

def process_dataset(dataset, classes, config_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['Network']
        
    config['batch_size'] = config.get('batch_size', 8)
    config['input_size'] = config.get('input_size', 256)
    glv.network_config = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = os.path.join(results_dir, f'grid_ablation_{dataset}.csv')
    txt_path = os.path.join(results_dir, f'grid_summary_{dataset}.txt')
    
    processed_configs = set()
    if os.path.exists(csv_path):
        print(f"Found existing results file at {csv_path}. Resuming...")
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    # key: Class_Mode_Timesteps_Alpha
                    k = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
                    processed_configs.add(k)
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Mode,Timesteps,Alpha,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,MAC(G),SOP(G),CalibTime(s),TestTime(s),FPS\n")
            
    for cls in classes:
        print(f"\n{'='*80}")
        print(f"[{dataset.upper()}] Processing Class: {cls}")
        print(f"{'='*80}")
        
        # 1. Load data ONCE per class (for SNN conversion)
        glv.network_config['batch_size'] = 8
        if dataset == 'mvtec':
            train_loader, test_loader = load_mvtec(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
        else:
            train_loader, test_loader = load_visa(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
            
        ann_encoder = BackboneEncoder(backbone=config['backbone'], layers=config['layers']).to(device)
        
        try:
            from thop import profile
            dummy = torch.randn(1, 3, config['input_size'], config['input_size']).to(device)
            macs, _ = profile(ann_encoder, inputs=(dummy,), verbose=False)
            macs_giga = macs / 1e9
        except ImportError:
            macs_giga = 0.0
            
        for mode in MODES:
            # Check if all configs for this class+mode are already done
            all_done = True
            for ts in TIMESTEPS:
                for a in ALPHAS:
                    if f"{cls}_{mode}_{ts}_{a}" not in processed_configs:
                        all_done = False
                        break
            
            if all_done:
                print(f"  [Mode={mode}] Skipping, already completely processed.")
                continue
                
            print(f"  [Mode={mode}] Building SNN Encoder...")
            t_b = time.time()
            snn_encoder = build_snn_encoder(ann_encoder, train_loader, device, mode=mode)
            build_time = time.time() - t_b
            
            for ts in TIMESTEPS:
                # Check if all alphas for this ts are done
                ts_done = True
                for a in ALPHAS:
                    if f"{cls}_{mode}_{ts}_{a}" not in processed_configs:
                        ts_done = False
                        break
                
                if ts_done:
                    continue
                    
                print(f"    [TS={ts}] Computing Normal Stats...")
                t_c = time.time()
                # We use a custom memory-safe normal stats computation here
                normal_stats = compute_normal_stats(snn_encoder, train_loader, device, ts, config['layers'])
                calib_time = build_time + (time.time() - t_c)
                
                # Cleanup state from Normal Stats
                functional.reset_net(snn_encoder)
                torch.cuda.empty_cache()
                
                print(f"    [TS={ts}] Running Fast Ablation on Test Set...")
                metrics_by_alpha, test_time, fps, fr_tb = evaluate_fast_ablation(
                    snn_encoder, test_loader, normal_stats, device, ts, config['layers'], 
                    config['input_size'], config['combine_method'], ALPHAS
                )
                
                sop_giga = macs_giga * fr_tb * ts
                
                # Save results
                with open(csv_path, 'a') as f:
                    for alpha in ALPHAS:
                        if f"{cls}_{mode}_{ts}_{alpha}" not in processed_configs:
                            m = metrics_by_alpha[alpha]
                            f.write(f"{cls},{mode},{ts},{alpha},{m['img_auc']:.4f},{m['img_ap']:.4f},{m['img_f1']:.4f},{m['pix_auc']:.4f},{m['pix_ap']:.4f},{m['pix_f1']:.4f},{m['pro']:.4f},{m['mad']:.4f},{macs_giga:.8f},{sop_giga:.8f},{calib_time:.2f},{test_time:.2f},{fps:.2f}\n")
                            processed_configs.add(f"{cls}_{mode}_{ts}_{alpha}")
                
                print(f"    [TS={ts}] Completed! Checkpoint saved.")
                
                main_s2ad._interpolators_cache.clear()
                del normal_stats
                torch.cuda.empty_cache()
                gc.collect()
                
            # Cleanup Mode loop
            del snn_encoder
            torch.cuda.empty_cache()
            gc.collect()
            
        # Cleanup Class loop
        del ann_encoder
        del train_loader
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    # Finally, generate the summary TXT
    summarize_results(csv_path, dataset, txt_path)
    print(f"\n[INFO] Grid search completed for {dataset.upper()}! Summary saved to: {txt_path}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Grid Search Ablation for S2AD (MVTec/VisA)")
    parser.add_argument('--dataset', type=str, default='both', choices=['mvtec', 'visa', 'both'])
    args = parser.parse_args()
    
    if args.dataset in ['mvtec', 'both']:
        process_dataset('mvtec', MVTEC_CLASSES, 'NetworkConfigs/s2ad_configs/MVTec.yaml', './results_grid_ablation_mvtec')
        
    if args.dataset in ['visa', 'both']:
        process_dataset('visa', VISA_CLASSES, 'NetworkConfigs/s2ad_configs/VisA.yaml', './results_grid_ablation_visa')
        
if __name__ == '__main__':
    main()
