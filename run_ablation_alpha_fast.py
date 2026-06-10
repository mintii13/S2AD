import os
import time
import argparse
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from main_s2ad import BackboneEncoder, build_snn_encoder, compute_normal_stats, get_firing_rates, get_zscore_layer, get_interpolator
from datasets.load_dataset_snn import load_mvtec, load_visa
import global_v as glv
from ad_eval import compute_pro_metric

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

ALPHAS = [0.0, 0.01, 0.05, 0.1, 0.2]

def evaluate_fast_ablation(snn_encoder, test_loader, normal_stats, device, timesteps, layers, img_size, combine_method, alphas):
    start_time = time.time()
    
    # Dictionaries to store results for each alpha
    results_by_alpha = {alpha: {'img_scores': [], 'img_labels': [], 'pix_scores': [], 'pix_labels': [], 'gt_masks': [], 'anomaly_maps': []} for alpha in alphas}
    
    for imgs, lbls, gt_paths in test_loader:
        imgs = imgs.to(device)
        # 1. FORWARD PASS MỘT LẦN DUY NHẤT (Nặng nhất)
        rates = get_firing_rates(snn_encoder, imgs, device, timesteps, layers)
        
        # 2. VÒNG LẶP QUA CÁC ALPHA TRÊN CÙNG MỘT TENSOR RATES (Cực nhẹ)
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
            
            # Lưu kết quả cho alpha này
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
                    
        # === QUAN TRỌNG: Giải phóng bộ nhớ NGAY LẬP TỨC sau mỗi batch ===
        from spikingjelly.activation_based import functional
        functional.reset_net(snn_encoder)
        del rates
        del imgs
        if 'deviations' in locals(): del deviations
        if 'score_spatial' in locals(): del score_spatial
        if 'score_maps' in locals(): del score_maps
        # torch.cuda.empty_cache() # Không nên dùng ở đây vì sẽ làm chậm, del và reset_net là đủ

    # Tính toán metrics cho từng alpha
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
        
    return final_metrics, time.time() - start_time

def process_dataset(dataset, classes, config_path, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['Network']
        
    config['batch_size'] = config.get('batch_size', 16)
    config['input_size'] = config.get('input_size', 256)
    glv.network_config = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = os.path.join(results_dir, f'ablation_alpha_fast_{dataset}.csv')
    
    processed_classes = set()
    if os.path.exists(csv_path):
        print(f"Found existing results file at {csv_path}. Resuming...")
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]: # Bỏ qua header
                parts = line.strip().split(',')
                if len(parts) > 0 and parts[0]:
                    processed_classes.add(parts[0])
    else:
        with open(csv_path, 'w') as f:
            f.write("Class,Alpha,mAD,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO\n")
        
    for cls in classes:
        if cls in processed_classes:
            print(f"\n{'='*60}")
            print(f"[{dataset.upper()}] Skipping Class (Already processed): {cls}")
            print(f"{'='*60}")
            continue
            
        print(f"\n{'='*60}")
        print(f"[{dataset.upper()}] Processing Class: {cls}")
        print(f"{'='*60}")
        
        # 1. Load data once
        if dataset == 'mvtec':
            train_loader, test_loader = load_mvtec(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
        else:
            train_loader, test_loader = load_visa(config['data_path'], cls, shuffle_train=False, drop_last_train=False, normalize='imagenet')
            
        # 2. Build model once
        ann_encoder = BackboneEncoder(backbone=config['backbone'], layers=config['layers']).to(device)
        snn_encoder = build_snn_encoder(ann_encoder, train_loader, device, mode=str(config.get('snn_mode', '0.4')))
        
        # 3. Compute Normal Stats once
        print("Computing Normal Stats...")
        normal_stats = compute_normal_stats(snn_encoder, train_loader, device, config['timesteps'][0], config['layers'])
        
        # Dọn dẹp trạng thái của mẻ Normal cuối cùng
        from spikingjelly.activation_based import functional
        functional.reset_net(snn_encoder)
        torch.cuda.empty_cache()
        
        # 4. Fast forward pass & evaluate all alphas
        print("Running Fast Ablation on Test Set...")
        metrics_by_alpha, test_time = evaluate_fast_ablation(
            snn_encoder, test_loader, normal_stats, device, config['timesteps'][0], config['layers'], 
            config['input_size'], config['combine_method'], ALPHAS
        )
        
        print(f"Testing completed in {test_time:.2f}s!")
        print(f"\n{'Alpha':>6} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8} | {'mAD':>8}")
        print('-' * 105)
        
        for alpha in ALPHAS:
            m = metrics_by_alpha[alpha]
            print(f"{alpha:>6.2f} | {m['img_auc']:8.4f} | {m['img_ap']:8.4f} | {m['img_f1']:8.4f} | {m['pix_auc']:8.4f} | {m['pix_ap']:8.4f} | {m['pix_f1']:8.4f} | {m['pro']:8.4f} | {m['mad']:8.4f}")
            with open(csv_path, 'a') as f:
                f.write(f"{cls},{alpha},{m['mad']},{m['img_auc']},{m['img_ap']},{m['img_f1']},{m['pix_auc']},{m['pix_ap']},{m['pix_f1']},{m['pro']}\n")

        # === QUAN TRỌNG: Dọn dẹp RAM & GPU sau mỗi class ===
        import gc
        import main_s2ad
        main_s2ad._zscore_cache.clear()
        main_s2ad._interpolators_cache.clear()
        
        del ann_encoder
        del snn_encoder
        del normal_stats
        del train_loader
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='both', choices=['mvtec', 'visa', 'both'])
    args = parser.parse_args()
    
    if args.dataset in ['mvtec', 'both']:
        process_dataset('mvtec', MVTEC_CLASSES, 'NetworkConfigs/s2ad_configs/MVTec.yaml', './results_ablation_alpha_mvtec')
        
    if args.dataset in ['visa', 'both']:
        process_dataset('visa', VISA_CLASSES, 'NetworkConfigs/s2ad_configs/VisA.yaml', './results_ablation_alpha_visa')
        
    print("\nDONE! Results saved to CSV files in the respective result directories.")

if __name__ == '__main__':
    main()
