import os
import glob
import argparse
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import ann2snn, functional, layer
from tqdm import tqdm
import gc

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def blend_map(score_map_np, orig_pil, alpha=0.4):
    smin, smax = score_map_np.min(), score_map_np.max()
    if smax > smin:
        score_norm = (score_map_np - smin) / (smax - smin)
    else:
        score_norm = score_map_np
    cmap = cm.jet(score_norm)
    anomaly_colored = (cmap[:, :, :3] * 255).astype(np.uint8)
    anomaly_pil = Image.fromarray(anomaly_colored)
    return Image.blend(orig_pil, anomaly_pil, alpha=alpha)

class BackboneEncoder(nn.Module):
    def __init__(self, backbone='vgg16'):
        super().__init__()
        self.is_resnet = False
        if backbone == 'alexnet':
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
            self.output_indices = [4, 7, 9]
        elif backbone == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.output_indices = [8, 15, 22]
        else:
            raise ValueError("Only supports alexnet or vgg16")
            
        self.features = model
        self.output_indices = sorted(self.output_indices)
    
    def forward(self, x):
        outputs = []
        for i, lyr in enumerate(self.features):
            x = lyr(x)
            if i in self.output_indices:
                outputs.append(x)
        while len(outputs) < 3: outputs.append(x)
        if len(outputs) > 3: outputs = outputs[:3]
        return tuple(outputs)

def build_snn_encoder_3l(ann_encoder, calib_loader, device, mode=0.9):
    ann_encoder.eval()
    class AdapterLoader:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for batch in self.loader: yield batch[0], batch[1]
        def __len__(self): return len(self.loader)
    
    converter = ann2snn.Converter(dataloader=AdapterLoader(calib_loader), device=device, mode=mode, momentum=0.1)
    snn_encoder = converter(ann_encoder)
    for module in snn_encoder.modules():
        if hasattr(module, 'output'): module.output = True
        if hasattr(module, 'out_spike'): module.out_spike = True

    def wrap_stateless(m):
        for name, child in m.named_children():
            if isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Linear)):
                setattr(m, name, layer.SeqToANNContainer(child))
            else: wrap_stateless(child)
    wrap_stateless(snn_encoder)
    functional.set_step_mode(snn_encoder, 'm')
    return snn_encoder

transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((256, 256)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(path):
    img = cv2.imread(path)
    if img is None: raise ValueError(f"Not found: {path}")
    return transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

class ImageDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx): return process_image(self.paths[idx]), torch.zeros(1)

def run_visualization_for_category(category, data_path, backbone, snn_mode, T, batch_size, device):
    print(f"\n[{category}] Đang khởi tạo mô hình và dữ liệu...")
    normal_dir = os.path.join(data_path, category, 'train', 'good')
    test_dir_base = os.path.join(data_path, category, 'test')
    gt_dir_base = os.path.join(data_path, category, 'ground_truth')
    
    out_dir = os.path.join('visualizations_s2ad', category)
    os.makedirs(out_dir, exist_ok=True)
    
    normal_paths = sorted(glob.glob(f"{normal_dir}/*.png"))
    if not normal_paths:
        print(f"Bỏ qua {category} vì không tìm thấy ảnh normal.")
        return
        
    calib_dataset = ImageDataset(normal_paths)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    normal_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)

    ann_model = BackboneEncoder(backbone=backbone).to(device)
    print(f"[{category}] Calibrate SNN...")
    snn_model = build_snn_encoder_3l(ann_model, calib_loader, device=device, mode=float(snn_mode))

    print(f"[{category}] Tính Normal Stats (Batch={batch_size})...")
    sum_rates = {l: None for l in ['layer1', 'layer2', 'layer3']}
    sum_sq_rates = {l: None for l in ['layer1', 'layer2', 'layer3']}
    max_rates = {l: 0.0 for l in ['layer1', 'layer2', 'layer3']}
    count = 0

    for imgs, _ in tqdm(normal_loader, desc="Pass 1", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        functional.reset_net(snn_model)
        with torch.no_grad():
            imgs_T = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            outputs = snn_model(imgs_T)
            for idx, l in zip([0, 1, 2], ['layer1', 'layer2', 'layer3']):
                rate = (outputs[idx] > 0).float().mean(dim=0)
                if sum_rates[l] is None:
                    sum_rates[l] = rate.sum(dim=0).cpu()
                    sum_sq_rates[l] = (rate ** 2).sum(dim=0).cpu()
                else:
                    sum_rates[l] += rate.sum(dim=0).cpu()
                    sum_sq_rates[l] += (rate ** 2).sum(dim=0).cpu()
                m_val = rate.max().item()
                if m_val > max_rates[l]: max_rates[l] = m_val
        count += B
        gc.collect()

    stats = {}
    for l in ['layer1', 'layer2', 'layer3']:
        mean_r = sum_rates[l] / count
        var_r = torch.clamp((sum_sq_rates[l] / count) - (mean_r ** 2), min=0.0)
        stats[l] = {'mean': mean_r.to(device), 'std': torch.sqrt(var_r + 1e-8).to(device), 'max_rate': max_rates[l]}

    print(f"[{category}] Tính MAD Weighting...")
    sum_abs_dev = {l: 0.0 for l in ['layer1', 'layer2', 'layer3']}
    count = 0
    for imgs, _ in tqdm(normal_loader, desc="Pass 2", leave=False):
        imgs = imgs.to(device)
        B = imgs.shape[0]
        functional.reset_net(snn_model)
        with torch.no_grad():
            imgs_T = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            outputs = snn_model(imgs_T)
            for idx, l in zip([0, 1, 2], ['layer1', 'layer2', 'layer3']):
                rate = (outputs[idx] > 0).float().mean(dim=0)
                abs_dev = torch.abs(rate - stats[l]['mean']).mean().item()
                sum_abs_dev[l] += abs_dev * B
        count += B
        gc.collect()

    for l in ['layer1', 'layer2', 'layer3']:
        stats[l]['mad'] = sum_abs_dev[l] / count

    print(f"\nTimestep T={T}:")
    for l in ['layer1', 'layer2', 'layer3']:
        mean_val = stats[l]['mean'].mean().item()
        max_val = stats[l]['max_rate']
        std_val = stats[l]['std'].mean().item()
        mad_val = stats[l]['mad']
        print(f"  {l}: mean={mean_val:.6f}, max={max_val:.6f}, std={std_val:.6f}, mAD={mad_val:.6f}")
    print()

    # GET ALL TEST IMAGES (excluding 'good' folder if desired, but we'll run all)
    test_folders = [d for d in os.listdir(test_dir_base) if os.path.isdir(os.path.join(test_dir_base, d))]
    
    for t_folder in test_folders:
        t_paths = sorted(glob.glob(os.path.join(test_dir_base, t_folder, '*.png')))
        print(f"[{category}] Đang vẽ {len(t_paths)} ảnh Test trong thư mục '{t_folder}'...")
        
        for image_path in tqdm(t_paths, desc=t_folder, leave=False):
            filename = os.path.basename(image_path)
            mask_name = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(gt_dir_base, t_folder, mask_name)
            
            if not os.path.exists(mask_path):
                mask_bin = np.zeros((256, 256), dtype=np.uint8)
            else:
                mask = cv2.resize(cv2.imread(mask_path, 0), (256, 256))
                mask_bin = (mask > 127).astype(np.uint8) * 255

            orig_img_cv = cv2.imread(image_path)
            orig_img_rgb = cv2.cvtColor(orig_img_cv, cv2.COLOR_BGR2RGB)
            orig_img_resized = cv2.resize(orig_img_rgb, (256, 256))
            img_pil = Image.fromarray(orig_img_resized)

            img_tensor_test = process_image(image_path).unsqueeze(0).to(device)
            functional.reset_net(snn_model)
            with torch.no_grad():
                test_imgs_T = img_tensor_test.unsqueeze(0).repeat(T, 1, 1, 1, 1)
                outputs = snn_model(test_imgs_T)
                test_rates = {l: (outputs[idx] > 0).float().mean(dim=0).squeeze(0) for idx, l in zip([0, 1, 2], ['layer1', 'layer2', 'layer3'])}

            maps = {}
            raw_maps = {'1_normal': {}, '2_test': {}, '3_abs_diff': {}, '4_zscore': {}}
            
            for l in ['layer1', 'layer2', 'layer3']:
                mean_r = stats[l]['mean'].squeeze(0)
                std_r = stats[l]['std'].squeeze(0)
                test_r = test_rates[l]
                
                raw_maps['1_normal'][l] = mean_r.mean(dim=0).unsqueeze(0).unsqueeze(0)
                raw_maps['2_test'][l] = test_r.mean(dim=0).unsqueeze(0).unsqueeze(0)
                raw_maps['3_abs_diff'][l] = torch.abs(test_r - mean_r).mean(dim=0).unsqueeze(0).unsqueeze(0)
                raw_maps['4_zscore'][l] = (torch.abs(test_r - mean_r) / std_r).mean(dim=0).unsqueeze(0).unsqueeze(0)

                maps[l] = {k: F.interpolate(raw_maps[k][l], size=(256, 256), mode='bilinear', align_corners=False).squeeze().cpu().numpy() for k in raw_maps.keys()}

            target_res = raw_maps['4_zscore']['layer1'].shape[2:] 
            maps['avg'] = {}
            maps['mad_weighted'] = {}

            for k in raw_maps.keys():
                avg_c = torch.zeros_like(raw_maps[k]['layer1'])
                for l in ['layer1', 'layer2', 'layer3']:
                    dev = raw_maps[k][l]
                    if dev.shape[2:] != target_res: dev = F.interpolate(dev, size=target_res, mode='bilinear', align_corners=False)
                    avg_c += dev
                avg_c /= 3.0
                maps['avg'][k] = F.interpolate(avg_c, size=(256, 256), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

                mad_c = torch.zeros_like(raw_maps[k]['layer1'])
                total_w = 0.0
                for l in ['layer1', 'layer2', 'layer3']:
                    dev = raw_maps[k][l]
                    if dev.shape[2:] != target_res: dev = F.interpolate(dev, size=target_res, mode='bilinear', align_corners=False)
                    weight = 1.0 / (stats[l]['mad'] + 1e-8)
                    mad_c += dev * weight
                    total_w += weight
                mad_c /= total_w
                maps['mad_weighted'][k] = F.interpolate(mad_c, size=(256, 256), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

            row_keys = ['layer1', 'layer2', 'layer3', 'avg', 'mad_weighted']
            for r_key in row_keys:
                maps[r_key]['5_blended'] = blend_map(maps[r_key]['4_zscore'], img_pil, alpha=0.4)
                maps[r_key]['6_input'] = img_pil 

            img_score = float(np.max(maps['mad_weighted']['4_zscore']))

            # Bảng 5x7 với Title to
            row_titles = ['Layer 1 (Shallow)', 'Layer 2 (Mid)', 'Layer 3 (Deep)', 'Average Combine', 'MAD Weighting']
            col_keys = ['1_normal', '2_test', '3_abs_diff', '4_zscore', '5_blended', '6_input']
            col_titles = ['Normal Firing Rate', 'Test Firing Rate', 'Abs Deviation Map', 'Z-Score Map', 'Blended Anomaly Map', 'Test Image', 'Ground Truth']

            import matplotlib.cm as cm
            from matplotlib.ticker import FuncFormatter
            
            # Format số đúng 4 chữ số (ví dụ: 0.123 hoặc 12.34)
            def custom_tick_format(val, pos):
                if val == 0: return "0.000"
                abs_v = abs(val)
                if abs_v >= 100: return f"{val:.1f}"
                elif abs_v >= 10: return f"{val:.2f}"
                else: return f"{val:.3f}"
            
            # Dùng 8 cột thay vì 7 cột. Chèn 1 cột rỗng (dummy) ở vị trí số 5 (index 4) để chủ động tạo khoảng cách giữa Z-score và Blended.
            # Cột 1-4 dùng 1.3 để cách đều nhau. Cột rỗng dùng 0.2 để giãn Z-score và Blended.
            # 3 cột cuối (Blended, Test, GT) dùng tỷ lệ 1.0 để dính chặt tuyệt đối.
            fig, axes = plt.subplots(5, 8, figsize=(47, 28), gridspec_kw={'width_ratios': [1.2, 1.2, 1.2, 1.2, 0.2, 1.0, 1.0, 1.0]}) 
            fig.suptitle(f"{category} | {t_folder}/{filename} ({backbone.upper()}) | Max Anomaly Score = {img_score:.4f}", fontsize=50, fontweight='bold', y=0.98, color='navy')

            for r, r_key in enumerate(row_keys):
                for c, c_key in enumerate(col_keys):
                    # Bỏ qua cột rỗng ở giữa (index 4)
                    ax_idx = c if c < 4 else c + 1
                    ax = axes[r, ax_idx]
                    if c_key in ['5_blended', '6_input']:
                        ax.imshow(maps[r_key][c_key]) 
                    else:
                        im = ax.imshow(maps[r_key][c_key], cmap='jet')
                        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        
                        # Lấy đúng 4 mốc, xích vào giữa 5% để chữ không thò ra ngoài viền trên/dưới
                        vmin, vmax = im.get_clim()
                        if vmax > vmin:
                            margin = (vmax - vmin) * 0.05
                            cb.set_ticks(np.linspace(vmin + margin, vmax - margin, 4))
                        else:
                            cb.set_ticks([vmin])
                            
                        # Format 4 chữ số
                        cb.ax.yaxis.set_major_formatter(FuncFormatter(custom_tick_format))
                        cb.ax.tick_params(labelsize=20)
                        
                    if r == 0: ax.set_title(col_titles[c], fontsize=30, fontweight='bold', pad=25)
                    if c == 0: ax.set_ylabel(row_titles[r], fontsize=30, fontweight='bold', labelpad=25)
                    ax.set_xticks([]); ax.set_yticks([])
                    
                # Tắt cột rỗng
                axes[r, 4].axis('off')
                    
                ax_gt = axes[r, 7]
                ax_gt.imshow(mask_bin, cmap='gray')
                if r == 0: ax_gt.set_title(col_titles[6], fontsize=30, fontweight='bold', pad=25)
                ax_gt.set_xticks([]); ax_gt.set_yticks([])

            # Thay thế tight_layout bằng subplots_adjust với wspace=0.0 để 3 ảnh cuối dính chặt hoàn toàn
            plt.subplots_adjust(wspace=0.0, hspace=0.1, left=0.02, right=0.98, top=0.92, bottom=0.02)
            
            save_name = f"{t_folder}_{filename}"
            save_path = os.path.join(out_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default=None, help='Class name (e.g. bottle). If not set, run all classes.')
    args = parser.parse_args()

    config_path = 'NetworkConfigs/s2ad_configs/MVTec.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['Network']
        
    data_path = config['data_path']
    backbone = config.get('backbone', 'vgg16')
    snn_mode = config.get('snn_mode', '0.9')
    batch_size = config.get('batch_size', 16)
    T = config.get('timesteps', [16])[-1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== S2AD Visualization Setup ===")
    print(f"Backbone: {backbone} | Mode: {snn_mode} | T: {T} | Batch: {batch_size}")
    
    if args.category is None:
        categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        print(f"Chạy tất cả {len(categories)} classes: {categories}")
        for cat in categories:
            run_visualization_for_category(cat, data_path, backbone, snn_mode, T, batch_size, device)
    else:
        run_visualization_for_category(args.category, data_path, backbone, snn_mode, T, batch_size, device)
        
    print("\n[✔] HOÀN TẤT VẼ BIỂU ĐỒ! Các file ảnh đã được lưu trong thư mục 'visualizations_s2ad/'")

if __name__ == '__main__':
    main()
