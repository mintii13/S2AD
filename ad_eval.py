import sys
import os
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc
from scipy.ndimage import label as connected_components
from PIL import Image
import matplotlib.cm as cm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def save_anomaly_map(original_img, score_map, gt_mask, save_dir, idx):
    img_pil = Image.fromarray(original_img)
    
    smin, smax = score_map.min(), score_map.max()
    if smax > smin:
        score_norm = (score_map - smin) / (smax - smin)
    else:
        score_norm = score_map
    cmap = cm.jet(score_norm)
    anomaly_colored = (cmap[:, :, :3] * 255).astype(np.uint8)
    anomaly_pil = Image.fromarray(anomaly_colored)
    
    os.makedirs(save_dir, exist_ok=True)
    img_pil.save(os.path.join(save_dir, f'{idx:04d}_img.png'))
    
    blended = Image.blend(img_pil, anomaly_pil, alpha=0.4)
    blended.save(os.path.join(save_dir, f'{idx:04d}_blend.png'))
    
    if gt_mask is not None:
        gt_pil = Image.fromarray(gt_mask).convert('RGB')
        gt_pil.save(os.path.join(save_dir, f'{idx:04d}_mask.png'))


def compute_pro_metric(gt_masks, anomaly_maps, fpr_limit=0.3):
    if not gt_masks or not anomaly_maps:
        return 0.0

    all_amaps = np.array(anomaly_maps)
    all_masks = np.array(gt_masks)

    normal_scores = all_amaps[all_masks == 0]
    total_normal_pixels = len(normal_scores)
    if total_normal_pixels == 0:
        return 0.0

    thresholds = np.linspace(all_amaps.min(), all_amaps.max(), 100)
    normal_scores_sorted = np.sort(normal_scores)

    regions_list = []
    for mask in all_masks:
        labeled, num_regions = connected_components(mask)
        regions_list.append([labeled == reg_id for reg_id in range(1, num_regions + 1)])

    fprs = []
    pros = []

    for t in thresholds:
        fp_count = total_normal_pixels - np.searchsorted(normal_scores_sorted, t)
        fpr = fp_count / total_normal_pixels
        fprs.append(fpr)

        overlaps = []
        for img_idx, regions in enumerate(regions_list):
            for region_mask in regions:
                region_scores = all_amaps[img_idx][region_mask]
                if region_scores.size > 0:
                    overlap_ratio = (region_scores >= t).sum() / region_scores.size
                    overlaps.append(overlap_ratio)

        pros.append(np.mean(overlaps) if overlaps else 0.0)

    fprs = np.array(fprs)
    pros = np.array(pros)

    idxes = fprs <= fpr_limit
    fprs_valid = fprs[idxes]
    pros_valid = pros[idxes]

    if len(fprs_valid) < 2:
        return 0.0

    fprs_normalized = (fprs_valid - fprs_valid.min()) / (fprs_valid.max() - fprs_valid.min() + 1e-8)
    pro_auc = auc(fprs_normalized, pros_valid)
    
    return float(pro_auc)

import time

def evaluate_ad(net, test_loader, device, epoch, args, dataset_name, cumulative_train_time=0.0, train_loss=0.0, test_loss=0.0):
    start_test_time = time.time()
    net.eval()
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
    gt_masks = []
    anomaly_maps = []
    
    import global_v as glv
    
    save_maps = glv.network_config.get('save_anomaly_maps', True)
    if save_maps:
        maps_dir = os.path.join(args.project_save_path, 'anomaly_maps', dataset_name, args.name)
        os.makedirs(maps_dir, exist_ok=True)
    else:
        maps_dir = None
    
    with torch.no_grad():
        for i, (img, lbl, gt_path) in enumerate(test_loader):
            img = img.to(device)
            spike_input = img.unsqueeze(-1).repeat(1, 1, 1, 1, glv.network_config['n_steps'])
            
            x_recon, r_q, r_p, sampled_z_q = net(spike_input, scheduled=True)
            
            mse = torch.mean((img - x_recon)**2, dim=1) # (B, H, W)
            
            for b in range(img.size(0)):
                score_map = mse[b].cpu().numpy()
                img_score = float(np.max(score_map))
                
                img_scores.append(img_score)
                label = lbl[b].item()
                img_labels.append(label)
                
                gt_p = gt_path[b] if isinstance(gt_path, (list, tuple)) else gt_path
                
                gt_mask = None
                if label == 1 and gt_p and os.path.exists(gt_p):
                    gt_mask = cv2.imread(gt_p, 0)
                    if gt_mask is not None:
                        gt_mask = cv2.resize(gt_mask, (score_map.shape[1], score_map.shape[0]))
                        gt_mask = (gt_mask > 127).astype(np.uint8) * 255
                        
                if maps_dir:
                    subfolder_name = 'good' if label == 0 else 'abnormal'
                    save_dir = os.path.join(maps_dir, subfolder_name)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    orig_img = (img[b].cpu() + 1) / 2.0
                    orig_img = orig_img.clamp(0,1).permute(1,2,0).numpy()
                    orig_img = (orig_img * 255).astype(np.uint8)
                    save_anomaly_map(orig_img, score_map, gt_mask, save_dir, i * img.size(0) + b)

                if label == 1 and gt_p:
                    if gt_mask is not None:
                        gt_bin = (gt_mask > 127).astype(int)
                        pix_scores.extend(score_map.flatten())
                        pix_labels.extend(gt_bin.flatten())
                        gt_masks.append(gt_bin)
                        anomaly_maps.append(score_map)
                        
    test_time = time.time() - start_test_time
    total_imgs = len(img_scores)
    fps = total_imgs / test_time if test_time > 0 else 0
                        
    img_auc = roc_auc_score(img_labels, img_scores) if len(set(img_labels)) == 2 else 0.0
    img_ap = average_precision_score(img_labels, img_scores) if len(set(img_labels)) == 2 else 0.0
    
    prec, rec, _ = precision_recall_curve(img_labels, img_scores)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    img_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
    
    pix_auc = roc_auc_score(pix_labels, pix_scores) if pix_labels else 0.0
    pix_ap = average_precision_score(pix_labels, pix_scores) if pix_labels else 0.0
    if pix_labels:
        pprec, prec_rec, _ = precision_recall_curve(pix_labels, pix_scores)
        pf1_scores = 2 * (pprec * prec_rec) / (pprec + prec_rec + 1e-8)
        pix_f1 = np.max(pf1_scores) if len(pf1_scores) > 0 else 0.0
    else:
        pix_f1 = 0.0
        
    pro_score = compute_pro_metric(gt_masks, anomaly_maps) if gt_masks else 0.0
    
    mAD_score = (img_auc + img_ap + img_f1 + pix_auc + pix_ap + pix_f1 + pro_score) / 7.0
    
    n_steps = glv.network_config['n_steps']
    out_path = os.path.join(args.project_save_path, f'{dataset_name}_{args.category}_T{n_steps}_ad_eval_results.txt')
    is_new_file = not os.path.exists(out_path)
    
    with open(out_path, 'a') as f:
        if is_new_file:
            f.write(f"\n{'Epoch':>8} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8} | {'mAD':>8} | {'TrainLoss':>9} | {'TestLoss':>9} | {'Train(s)':>8} | {'Test(s)':>8} | {'FPS':>8}\n")
            f.write('-' * 155 + '\n')
            
        row = f'{epoch:8d} | {img_auc:8.4f} | {img_ap:8.4f} | {img_f1:8.4f} | {pix_auc:8.4f} | {pix_ap:8.4f} | {pix_f1:8.4f} | {pro_score:8.4f} | {mAD_score:8.4f} | {train_loss:9.4f} | {test_loss:9.4f} | {cumulative_train_time:8.1f} | {test_time:8.2f} | {fps:8.2f}'
        f.write(row + '\n')
        print(row)
        
    return {'img_auc': img_auc, 'img_ap': img_ap, 'img_f1': img_f1, 'pix_auc': pix_auc, 'pix_ap': pix_ap, 'pix_f1': pix_f1, 'pro': pro_score, 'mad_metric': mAD_score}
