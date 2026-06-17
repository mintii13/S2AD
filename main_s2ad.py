import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models
import random
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from spikingjelly.activation_based import ann2snn, functional, layer
import setproctitle

import global_v as glv
from network_parser import parse
from datasets.load_dataset_snn import load_mvtec, load_visa
from ad_eval import save_anomaly_map, compute_pro_metric

setproctitle.setproctitle("python_s2ad")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 - Backbone Encoder
# ═══════════════════════════════════════════════════════════════════════════

class BackboneEncoder(nn.Module):
    def __init__(self, backbone='resnet18', layers='layer23'):
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self._build_backbone(backbone)
        
    def _build_backbone(self, backbone):
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2']:
            if backbone == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif backbone == 'resnet34':
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif backbone == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            elif backbone == 'wide_resnet50_2':
                model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            elif backbone == 'wide_resnet101_2':
                model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.is_resnet = True
            return
        
        self.is_resnet = False
        if backbone.startswith('vgg'):
            variants = {'vgg11': (models.vgg11, 8, 15, 22), 'vgg13': (models.vgg13, 6, 11, 16),
                        'vgg16': (models.vgg16, 8, 15, 22), 'vgg19': (models.vgg19, 9, 16, 25)}
            creator, idx1, idx2, idx3 = variants[backbone]
            model = creator(weights='IMAGENET1K_V1').features
            self.output_indices = [idx1, idx2, idx3]
        elif backbone == 'alexnet':
            model = models.alexnet(weights='IMAGENET1K_V1').features
            self.output_indices = [4, 7, 9]
        elif backbone == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            self.output_indices = [3, 10, 17]
        elif backbone == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1).features
            self.output_indices = [3, 8, 12]
        elif backbone in ['densenet121', 'densenet169']:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features if backbone == 'densenet121' else models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features
            self.output_indices = [4, 6, 8]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.features = model
        self.output_indices = sorted(self.output_indices)
    
    def forward(self, x):
        if self.is_resnet:
            x = self.stem(x)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            outputs = [f1, f2, f3]
        else:
            outputs = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.output_indices:
                    outputs.append(x)
            while len(outputs) < 3: outputs.append(x)
            if len(outputs) > 3: outputs = outputs[:3]
        
        if self.layers == 'layer1': return (outputs[0],)
        elif self.layers == 'layer2': return (outputs[1],)
        elif self.layers == 'layer3': return (outputs[2],)
        elif self.layers == 'layer12': return (outputs[0], outputs[1])
        elif self.layers == 'layer23': return (outputs[1], outputs[2])
        elif self.layers == 'layer123': return tuple(outputs)
        else: return (outputs[1], outputs[2])

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 - SNN Conversion
# ═══════════════════════════════════════════════════════════════════════════

def build_snn_encoder(ann_encoder, calib_loader, device, mode='max'):
    ann_encoder.eval()
    
    class AdapterLoader:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for batch in self.loader: yield batch[0], batch[1]
        def __len__(self): return len(self.loader)
    
    adapter = AdapterLoader(calib_loader)
    converter_mode = mode if mode == 'max' else float(mode)
    
    converter = ann2snn.Converter(dataloader=adapter, device=device, mode=converter_mode, momentum=0.1)
    snn_encoder = converter(ann_encoder)
    for module in snn_encoder.modules():
        if hasattr(module, 'output'): module.output = True
        if hasattr(module, 'out_spike'): module.out_spike = True

    def wrap_stateless(m):
        for name, child in m.named_children():
            if isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Linear)):
                setattr(m, name, layer.SeqToANNContainer(child))
            else:
                wrap_stateless(child)
                
    wrap_stateless(snn_encoder)
    functional.set_step_mode(snn_encoder, 'm')

    print(f"  ANN2SNN conversion complete (mode={converter_mode}, wrapped for Multi-step)")
    return snn_encoder

def get_layer_indices_and_names(layers):
    mapping = {
        'layer1': ([0], ['layer1']), 'layer2': ([0], ['layer2']), 'layer3': ([0], ['layer3']),
        'layer12': ([0, 1], ['layer1', 'layer2']), 'layer23': ([0, 1], ['layer2', 'layer3']),
        'layer13': ([0, 2], ['layer1', 'layer3']),
        'layer123': ([0, 1, 2], ['layer1', 'layer2', 'layer3']),
    }
    return mapping.get(layers, ([0, 1], ['layer2', 'layer3']))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 - Normal Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_normal_stats(snn_encoder, normal_loader, device, timesteps, layers='layer23'):
    snn_encoder.eval()
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    
    sum_rates = {name: None for name in layer_names}
    sum_sq_rates = {name: None for name in layer_names}
    max_rates = {name: 0.0 for name in layer_names}
    count = 0
    
    with torch.no_grad():
        for imgs, _, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            chunk_size = max(1, 16 * 16 // timesteps)
            
            for i in range(0, B, chunk_size):
                chunk = imgs[i:i+chunk_size]
                chunk_B = chunk.shape[0]
                functional.reset_net(snn_encoder)
                
                imgs_T = chunk.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)
                outputs = snn_encoder(imgs_T)
                
                for idx, name in zip(layer_indices, layer_names):
                    feat = outputs[idx]
                    spike = (feat > 0).float()
                    rate = spike.mean(dim=0)
                    if sum_rates[name] is None:
                        sum_rates[name] = rate.sum(dim=0).cpu()
                        sum_sq_rates[name] = (rate ** 2).sum(dim=0).cpu()
                    else:
                        sum_rates[name] += rate.sum(dim=0).cpu()
                        sum_sq_rates[name] += (rate ** 2).sum(dim=0).cpu()
                    current_max = rate.max().item()
                    if current_max > max_rates[name]: max_rates[name] = current_max
                count += chunk_B
                
                # === CLEANUP MEMORY ===
                functional.reset_net(snn_encoder)
                del imgs_T
                del outputs
                del feat
                del spike
                del rate
            
    means, stats = {}, {}
    for name in layer_names:
        mean = sum_rates[name] / count
        var = torch.clamp((sum_sq_rates[name] / count) - (mean ** 2), min=0.0)
        std = torch.sqrt(var + 1e-8)
        means[name] = mean
        stats[name] = {'mean': mean, 'std': std, 'max_rate': max_rates[name]}
    
    sum_abs_dev = {name: 0.0 for name in layer_names}
    count = 0
    torch.cuda.empty_cache() # Dọn dẹp trước loop 2
    
    with torch.no_grad():
        for imgs, _, _ in normal_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            chunk_size = max(1, 16 * 16 // timesteps)
            
            for i in range(0, B, chunk_size):
                chunk = imgs[i:i+chunk_size]
                chunk_B = chunk.shape[0]
                functional.reset_net(snn_encoder)
                
                imgs_T = chunk.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)
                outputs = snn_encoder(imgs_T)
                
                for idx, name in zip(layer_indices, layer_names):
                    feat = outputs[idx]
                    spike = (feat > 0).float()
                    rate = spike.mean(dim=0)
                    abs_dev = torch.abs(rate - means[name].to(device)).mean().item()
                    sum_abs_dev[name] += abs_dev * chunk_B
                count += chunk_B
                
                # === CLEANUP MEMORY ===
                functional.reset_net(snn_encoder)
                del imgs_T
                del outputs
                del feat
                del spike
                del rate
            
    torch.cuda.empty_cache() # Dọn dẹp sau loop 2
    
    for name in layer_names:
        mad = sum_abs_dev[name] / count
        stats[name]['mad'] = mad
        print(f'    {name}: mean={stats[name]["mean"].mean().item():.4f}, max_rate={stats[name]["max_rate"]:.4f}, std={stats[name]["std"].mean().item():.4f}, mad={mad:.6f}')
    return stats

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 - Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def get_firing_rates(snn_encoder, img_tensor, device, timesteps, layers='layer23'):
    layer_indices, layer_names = get_layer_indices_and_names(layers)
    rates = {}
    with torch.no_grad():
        B = img_tensor.shape[0]
        chunk_size = max(1, 16 * 16 // timesteps)
        
        for i in range(0, B, chunk_size):
            chunk = img_tensor[i:i+chunk_size]
            functional.reset_net(snn_encoder)
            
            imgs_T = chunk.unsqueeze(0).repeat(timesteps, 1, 1, 1, 1)
            outputs = snn_encoder(imgs_T)
            
            for idx, name in zip(layer_indices, layer_names):
                feat = outputs[idx]
                spike = (feat > 0).float()
                rate_chunk = spike.mean(dim=0)
                
                if name not in rates:
                    rates[name] = []
                rates[name].append(rate_chunk)
                
            functional.reset_net(snn_encoder)
            del imgs_T
            del outputs
            del feat
            del spike
            del rate_chunk
            
        for name in layer_names:
            rates[name] = torch.cat(rates[name], dim=0)
            
    return rates

def _get_membrane_score(snn_encoder):
    for name, module in snn_encoder.named_modules():
        if 'IFNode' in str(type(module)) and hasattr(module, 'v') and module.v is not None:
            return module.v.pow(2).mean(dim=1).squeeze(0)
    return None

class HardwareFriendlyZScoreAbs(nn.Module):
    def __init__(self, mean, std, use_zscore=True, alpha=0.0):
        super().__init__()
        self.C, self.H, self.W = mean.shape
        self.alpha = alpha
        
        if use_zscore:
            if alpha == 0.0:
                smooth_std = std + 1e-8
            else:
                max_std = std.max()
                clamp_val = max(max_std * alpha, 1e-4)
                smooth_std = torch.clamp(std, min=clamp_val)
            self.register_buffer('w', 1.0 / smooth_std) # (C, H, W)
        else:
            self.register_buffer('w', torch.ones_like(std))
            
        self.register_buffer('b', -mean * self.w)

    def forward(self, x):
        # 1. Spatially unshared Z-score
        z = x * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        
        # 2. Dual Neuron Absolute Value
        # Positive stream
        z_pos = F.relu(z)
        # Negative stream (inhibitory weight)
        z_neg = F.relu(-z)
        
        # 3. Sum over channels (synaptic integration)
        # out = 1/C * sum(z_pos + z_neg)
        z_abs = z_pos + z_neg
        out = z_abs.mean(dim=1)
        
        return out

class HardwareFriendlyInterpolator(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        H_in, W_in = in_shape
        H_out, W_out = out_shape
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        if in_shape != out_shape:
            # Biến phép nội suy thành các kết nối Neuromorphic Synapse (Linear Layer)
            self.linear = nn.Linear(H_in * W_in, H_out * W_out, bias=False)
            
            # Dùng ma trận đơn vị để "trích xuất" chính xác bộ trọng số Bilinear
            with torch.no_grad():
                identity = torch.eye(H_in * W_in).view(H_in * W_in, 1, H_in, W_in)
                mapped = F.interpolate(identity, size=(H_out, W_out), mode='bilinear', align_corners=False)
                W_matrix = mapped.view(H_in * W_in, H_out * W_out)
                self.linear.weight.data = W_matrix.t() # (out_features, in_features)

    def forward(self, x):
        if self.in_shape == self.out_shape:
            return x
        B = x.shape[0]
        x_flat = x.view(B, -1)
        y_flat = self.linear(x_flat)
        return y_flat.view(B, self.out_shape[0], self.out_shape[1])

_interpolators_cache = {}
def get_interpolator(in_shape, out_shape, device):
    key = (in_shape, out_shape)
    if key not in _interpolators_cache:
        interpolator = HardwareFriendlyInterpolator(in_shape, out_shape).to(device)
        interpolator.eval()
        _interpolators_cache[key] = interpolator
    return _interpolators_cache[key]

def get_zscore_layer(layer_name, normal_stats, device, use_zscore, alpha=0.0):
    mean = normal_stats[layer_name]['mean'].to(device)
    std = normal_stats[layer_name]['std'].to(device)
    hw_layer = HardwareFriendlyZScoreAbs(mean, std, use_zscore=use_zscore, alpha=alpha).to(device)
    hw_layer.eval()
    return hw_layer

def score_image_batch(snn_encoder, img_tensor, normal_stats, device, timesteps, layers='layer23', img_size=256, use_membrane=False, combine_method='simple', use_zscore=True, alpha=0.0):
    snn_encoder.eval()
    img_tensor = img_tensor.to(device)
    rates = get_firing_rates(snn_encoder, img_tensor, device, timesteps, layers)
    
    deviations = {}
    for layer_name, rate in rates.items():
        hw_layer = get_zscore_layer(layer_name, normal_stats, device, use_zscore, alpha)
        with torch.no_grad():
            deviations[layer_name] = hw_layer(rate)
    
    if len(deviations) == 1:
        score_spatial = list(deviations.values())[0]
    else:
        target_name = list(deviations.keys())[0]
        target_res = deviations[target_name].shape[1:]
        if combine_method == 'simple':
            combined = torch.zeros_like(deviations[target_name])
            for layer_name, dev in deviations.items():
                if dev.shape[1:] != target_res:
                    interpolator = get_interpolator(dev.shape[1:], target_res, device)
                    with torch.no_grad():
                        dev = interpolator(dev)
                combined += dev
            score_spatial = combined / len(deviations)
        else:
            weighted_sum, total_weight = None, 0.0
            for layer_name, dev in deviations.items():
                if dev.shape[1:] != target_res:
                    interpolator = get_interpolator(dev.shape[1:], target_res, device)
                    with torch.no_grad():
                        dev = interpolator(dev)
                weight = 1.0 / (normal_stats[layer_name]['mad'] + 1e-8)
                total_weight += weight
                if weighted_sum is None: weighted_sum = dev * weight
                else: weighted_sum += dev * weight
            score_spatial = weighted_sum / total_weight
            
    final_interpolator = get_interpolator(score_spatial.shape[1:], (img_size, img_size), device)
    with torch.no_grad():
        score_maps = final_interpolator(score_spatial).cpu().numpy()
    img_scores = [float(np.max(sm)) for sm in score_maps]
    return score_maps, img_scores

def evaluate(snn_encoder, test_loader, normal_stats, device, timesteps, layers, img_size, use_membrane, combine_method, save_maps, maps_dir, category_name, use_zscore=True, alpha=0.0):
    """Evaluate using the EXACT same scoring logic as evaluate_fast_ablation, single pass, with optional map saving."""
    import cv2
    import time
    from spikingjelly.activation_based import functional
    
    start_time = time.time()
    total_save_time = 0.0
    img_scores, img_labels, pix_scores, pix_labels, gt_masks, anomaly_maps_list = [], [], [], [], [], []
    img_idx = 0
    
    if save_maps and maps_dir:
        from ad_eval import IMAGENET_MEAN, IMAGENET_STD
        mean_t, std_t = torch.tensor(IMAGENET_MEAN).view(3,1,1), torch.tensor(IMAGENET_STD).view(3,1,1)
    
    for imgs, lbls, gt_paths in test_loader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        
        # === Scoring logic identical to evaluate_fast_ablation ===
        rates = get_firing_rates(snn_encoder, imgs, device, timesteps, layers)
        
        deviations = {}
        for layer_name, rate in rates.items():
            hw_layer = get_zscore_layer(layer_name, normal_stats, device, use_zscore=use_zscore, alpha=alpha)
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
        # === End scoring logic ===
        
        for b in range(B):
            score_map = score_maps[b]
            lbl = lbls[b].item()
            gt_path = gt_paths[b]
            
            img_scores.append(batch_img_scores[b])
            img_labels.append(lbl)
            
            # Save anomaly map if requested (excluded from test_time)
            if save_maps and maps_dir:
                t_save = time.time()
                subfolder_name = 'abnormal' if lbl == 1 else 'good'
                save_dir = os.path.join(maps_dir, subfolder_name)
                os.makedirs(save_dir, exist_ok=True)
                orig_img = (imgs[b].cpu() * std_t + mean_t).clamp(0,1).permute(1,2,0).numpy()
                orig_img = (orig_img * 255).astype(np.uint8)
                
                gt_mask_vis = None
                if lbl == 1 and gt_path and os.path.exists(gt_path):
                    gt_mask_vis = cv2.resize(cv2.imread(gt_path, 0), (img_size, img_size))
                    gt_mask_vis = (gt_mask_vis > 127).astype(np.uint8) * 255
                save_anomaly_map(orig_img, score_map, gt_mask_vis, save_dir, img_idx)
                img_idx += 1
                total_save_time += time.time() - t_save
            
            # Collect for PRO metric
            if lbl == 1 and gt_path and os.path.exists(gt_path):
                gt = cv2.resize(cv2.imread(gt_path, 0), (img_size, img_size))
                gt_bin = (gt > 127).astype(int)
                pix_scores.extend(score_map.flatten())
                pix_labels.extend(gt_bin.flatten())
                gt_masks.append(gt_bin)
                anomaly_maps_list.append(score_map)
        
        # Cleanup per batch
        functional.reset_net(snn_encoder)
        del rates, deviations, score_maps
        torch.cuda.empty_cache()
    
    test_time = time.time() - start_time - total_save_time
    
    img_auc = roc_auc_score(img_labels, img_scores) if len(set(img_labels)) == 2 else 0.0
    img_ap = average_precision_score(img_labels, img_scores) if len(set(img_labels)) == 2 else 0.0
    prec, rec, _ = precision_recall_curve(img_labels, img_scores)
    img_f1 = np.max(2 * (prec * rec) / (prec + rec + 1e-8)) if len(prec) > 0 else 0.0
    
    pix_auc = roc_auc_score(pix_labels, pix_scores) if pix_labels else 0.0
    pix_ap = average_precision_score(pix_labels, pix_scores) if pix_labels else 0.0
    if pix_labels:
        pprec, prec_rec, _ = precision_recall_curve(pix_labels, pix_scores)
        pix_f1 = np.max(2 * (pprec * prec_rec) / (pprec + prec_rec + 1e-8)) if len(pprec) > 0 else 0.0
    else:
        pix_f1 = 0.0
    pro_score = compute_pro_metric(gt_masks, anomaly_maps_list) if gt_masks else 0.0
    
    return {'img_auc': img_auc, 'img_ap': img_ap, 'img_f1': img_f1, 'pix_auc': pix_auc, 'pix_ap': pix_ap, 'pix_f1': pix_f1, 'pro': pro_score, 'test_time': test_time}, img_scores, img_labels

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 - Main
# ═══════════════════════════════════════════════════════════════════════════

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, required=True, help='Category name or Exp name')
    parser.add_argument('-category', type=str, default='', help='Category name (for compatibility with run_all scripts)')
    parser.add_argument('-config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('-project_save_path', type=str, default='./results_s2ad')
    parser.add_argument('-alpha', type=float, default=None, help='Soft Z-score alpha')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    g = torch.Generator(); g.manual_seed(args.seed)
    
    print("\n" + "="*50)
    print(f"🚀 THÔNG BÁO TỪ HỆ THỐNG: ĐANG DÙNG SEED = {args.seed} 🚀")
    print("="*50 + "\n")
    
    category_name = args.category if args.category else args.name

    config = parse(args.config)['Network']
    config['batch_size'] = config.get('batch_size', 8)
    config['input_size'] = config.get('input_size', 256)
    glv.network_config = config
    os.makedirs(args.project_save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net_config = glv.network_config
    dataset_name = net_config['dataset']
    backbone = net_config.get('backbone', 'resnet18')
    layers = net_config.get('layers', 'layer23')
    timesteps = net_config.get('timesteps', [16])
    use_membrane = net_config.get('use_membrane', False)
    snn_mode = net_config.get('snn_mode', 'max')
    calib_samples = net_config.get('calib_samples', 500)
    combine_method = net_config.get('combine_method', 'mad_weighted')
    save_maps = net_config.get('save_anomaly_maps', False)
    use_zscore = net_config.get('use_zscore', True)
    img_size = net_config['input_size']
    alpha = args.alpha if args.alpha is not None else net_config.get('soft_zscore_alpha', 0.0)
    
    print('=' * 60)
    print(f'S2AD - Statistical SNN-based Anomaly Detection')
    print(f'  Category: {category_name} ({dataset_name})')
    print(f'  Backbone: {backbone} | Layers: {layers}')
    print(f'  Timesteps: {timesteps} | Soft Z-score Alpha: {alpha}')
    print('=' * 60)
    
    print('\n[1/4] Building encoder...')
    ann_encoder = BackboneEncoder(backbone=backbone, layers=layers).to(device)
    
    print('\n[2/4] Loading normal dataset...')
    if dataset_name == 'mvtec':
        train_loader, test_loader = load_mvtec(net_config['data_path'], category_name, shuffle_train=False, drop_last_train=False, normalize='imagenet')
    else:
        train_loader, test_loader = load_visa(net_config['data_path'], category_name, shuffle_train=False, drop_last_train=False, normalize='imagenet')
        
    full_normal_ds = train_loader.dataset
    if 0 < calib_samples < len(full_normal_ds):
        calib_loader = DataLoader(Subset(full_normal_ds, list(range(calib_samples))), batch_size=net_config['batch_size'], shuffle=False, num_workers=2, generator=g)
    else:
        calib_loader = train_loader
        
    print(f'  Normal set: {len(full_normal_ds)} images')
    
    print('\n[3/4] Converting ANN to SNN...')
    snn_encoder = build_snn_encoder(ann_encoder, calib_loader, device, mode=snn_mode)
    
    import time
    print('\n[4/4] Evaluating across timesteps...')
    print(f'\n{"Timestep":>8} | {"Img AUC":>8} | {"Img AP":>8} | {"Img F1":>8} | {"Pix AUC":>8} | {"Pix AP":>8} | {"Pix F1":>8} | {"PRO":>8} | {"mAD":>8} | {"Train(s)":>8} | {"Test(s)":>7} | {"FPS":>7}')
    print('-' * 135)
    
    # CUDA Warmup để tránh việc T=4 bị gánh overhead 10s khởi tạo
    # print("  [CUDA Warmup] Initializing cuDNN and allocating cache...")
    # from spikingjelly.clock_driven import functional
    # snn_encoder.eval()
    # with torch.no_grad():
    #     dummy_input = torch.randn(2, 3, img_size, img_size).to(device)
    #     dummy_spike = dummy_input.unsqueeze(0).repeat(max(timesteps), 1, 1, 1, 1)
    #     _ = snn_encoder(dummy_spike)
    #     functional.reset_net(snn_encoder)
        
    #     # Warmup Interpolators (cái này tốn 10s vì khởi tạo nn.Linear hàng triệu params)
    #     _ = get_interpolator((128, 128), (img_size, img_size), device)
    #     _ = get_interpolator((64, 64), (img_size, img_size), device)
    #     _ = get_interpolator((32, 32), (img_size, img_size), device)
        
    # torch.cuda.empty_cache()
    # import gc; gc.collect()

    results = {}
    firing_rate_stats = {}
    for T in timesteps:
        start_train = time.time()
        normal_stats = compute_normal_stats(snn_encoder, calib_loader, device, T, layers)
        train_time = time.time() - start_train
        
        firing_rate_stats[T] = {}
        for name, stats in normal_stats.items():
            firing_rate_stats[T][name] = {
                'mean': stats['mean'].mean().item(),
                'std': stats['std'].mean().item(),
                'max': stats['max_rate'],
                'mad': stats['mad']
            }
            
        maps_dir = os.path.join(args.project_save_path, 'anomaly_maps', f"{backbone}_{combine_method}{layers}_zscore{use_zscore}_alpha{alpha}", category_name, f"T{T}") if save_maps else None
        
        metrics, _, _ = evaluate(snn_encoder, test_loader, normal_stats, device, T, layers, img_size, use_membrane, combine_method, save_maps, maps_dir, category_name, use_zscore, alpha)
        test_time = metrics['test_time']
        fps = len(test_loader.dataset) / test_time if test_time > 0 else 0
        
        mAD_score = (metrics["img_auc"] + metrics["img_ap"] + metrics["img_f1"] + metrics["pix_auc"] + metrics["pix_ap"] + metrics["pix_f1"] + metrics["pro"]) / 7.0
        metrics["mad_metric"] = mAD_score
        metrics["train_time"] = train_time
        metrics["test_time"] = test_time
        metrics["fps"] = fps
        results[T] = metrics
        print(f'{T:8d} | {metrics["img_auc"]:8.4f} | {metrics["img_ap"]:8.4f} | {metrics["img_f1"]:8.4f} | {metrics["pix_auc"]:8.4f} | {metrics["pix_ap"]:8.4f} | {metrics["pix_f1"]:8.4f} | {metrics["pro"]:8.4f} | {metrics["mad_metric"]:8.4f} | {metrics["train_time"]:8.1f} | {metrics["test_time"]:7.1f} | {metrics["fps"]:7.1f}')
        
        ad_eval_out_path = os.path.join(args.project_save_path, f'{dataset_name}_{category_name}_T{T}_ad_eval_results.txt')
        is_new_file = not os.path.exists(ad_eval_out_path)
        with open(ad_eval_out_path, 'a') as fa:
            if is_new_file:
                fa.write(f"\n{'Epoch':>8} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8} | {'mAD':>8} | {'TrainLoss':>9} | {'TestLoss':>9} | {'Train(s)':>8} | {'Test(s)':>8} | {'FPS':>8}\n")
                fa.write('-' * 155 + '\n')
            fa.write(f'{199:8d} | {metrics["img_auc"]:8.4f} | {metrics["img_ap"]:8.4f} | {metrics["img_f1"]:8.4f} | {metrics["pix_auc"]:8.4f} | {metrics["pix_ap"]:8.4f} | {metrics["pix_f1"]:8.4f} | {metrics["pro"]:8.4f} | {metrics["mad_metric"]:8.4f} | {0.0:9.4f} | {0.0:9.4f} | {train_time:8.1f} | {test_time:8.2f} | {fps:8.2f}\n')
            
        del normal_stats
        del metrics
        import gc
        gc.collect()
    out_path = os.path.join(args.project_save_path, f'{category_name}_s2ad_results.txt')
    with open(out_path, 'w') as f:
        f.write(f"S2AD Results - {category_name}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Backbone: {backbone} | Layers: {layers}\n")
        f.write(f"Combine Method: {combine_method}\n")
        f.write(f"SNN Mode: {snn_mode}\n")
        f.write(f"\n{'Timestep':>8} | {'Img AUC':>8} | {'Img AP':>8} | {'Img F1':>8} | {'Pix AUC':>8} | {'Pix AP':>8} | {'Pix F1':>8} | {'PRO':>8} | {'mAD':>8} | {'Train(s)':>8} | {'Test(s)':>7} | {'FPS':>7}\n")
        f.write('-' * 135 + '\n')
        for T in sorted(results.keys()):
            f.write(f'{T:8d} | {results[T]["img_auc"]:8.4f} | {results[T]["img_ap"]:8.4f} | {results[T]["img_f1"]:8.4f} | {results[T]["pix_auc"]:8.4f} | {results[T]["pix_ap"]:8.4f} | {results[T]["pix_f1"]:8.4f} | {results[T]["pro"]:8.4f} | {results[T]["mad_metric"]:8.4f} | {results[T]["train_time"]:8.1f} | {results[T]["test_time"]:7.1f} | {results[T]["fps"]:7.1f}\n')
            
        f.write(f"\n\nFiring Rate Statistics:\n")
        f.write(f"{'=' * 50}\n")
        for T in sorted(firing_rate_stats.keys()):
            f.write(f"\nTimestep T={T}:\n")
            for layer_name, stats in firing_rate_stats[T].items():
                f.write(f"  {layer_name}: mean={stats['mean']:.6f}, max={stats['max']:.6f}, std={stats['std']:.6f}, mAD={stats['mad']:.6f}\n")
    print(f'\nResults saved: {out_path}')
if __name__ == '__main__':
    main()