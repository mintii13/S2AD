import torch
import os
import sys
import pandas as pd
from run_grid_ablation import MVTEC_CLASSES

sys.path.append('/home/minhtringuyen/ANN2SNN')
from main_s2ad import build_snn_encoder, compute_normal_stats, BackboneEncoder
from datasets.load_dataset_snn import load_mvtec

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("⏳ Đang đo đạc Firing Rate và SOPs CHÍNH XÁC cho bảng Mode Ablation...")
    
    T_list = [4, 8, 16, 32, 64]
    mode_list = ['1.0', '0.8', '0.6', '0.4', '0.2']
    
    results_dir = "./results_hardware_footprint"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "mode_ablation_sops.csv")
    progress_path = os.path.join(results_dir, "mode_ablation_progress.csv")
    
    processed = {}
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) == 5:
                    mode, T_val, cls, t_mean, t_elements = parts
                    key = f"{mode}_{T_val}_{cls}"
                    processed[key] = (float(t_mean), float(t_elements))
                    
    ann_enc = BackboneEncoder(backbone='vgg16', layers='layer123').to(device)
    
    VGG_MACS = 20.044578816
    POST_PARAMS = 0.003702784
    TOTAL_OPS = VGG_MACS + POST_PARAMS
    
    results_out = []
    
    for mode in mode_list:
        for T in T_list:
            global_total_mean = 0.0
            global_total_elements = 0.0
            
            for idx, cls in enumerate(MVTEC_CLASSES):
                key = f"{mode}_{T}_{cls}"
                if key in processed:
                    t_mean, t_elements = processed[key]
                    global_total_mean += t_mean
                    global_total_elements += t_elements
                    continue
                    
                data_path = '/home/minhtringuyen/ANN2SNN/dataloader/datasets/mvtec'
                train_loader, _ = load_mvtec(data_path, cls, batch_size=8, input_size=256)
                
                snn_encoder = build_snn_encoder(ann_enc, train_loader, device, mode=mode)
                stats = compute_normal_stats(snn_encoder, train_loader, device, T, 'layer123')
                
                cls_total_mean = 0.0
                cls_total_elements = 0.0
                for name, stat in stats.items():
                    mean_val = stat['mean'].mean().item()
                    num_elements = stat['mean'].numel()
                    cls_total_mean += mean_val * num_elements
                    cls_total_elements += num_elements
                    
                with open(progress_path, 'a') as f:
                    f.write(f"{mode},{T},{cls},{cls_total_mean},{cls_total_elements}\n")
                    
                global_total_mean += cls_total_mean
                global_total_elements += cls_total_elements
                
                print(f"  [Mode {mode}] T={T} | {cls} Xong.")
                
            avg_firing_rate = global_total_mean / global_total_elements
            sops = TOTAL_OPS * T * avg_firing_rate
            
            print(f"✅ Mode {mode} | T={T} | Firing Rate: {avg_firing_rate:.6f} | SOPs: {sops:.2f} G")
            
            results_out.append({
                'Mode': mode,
                'T': T,
                'FiringRate': avg_firing_rate,
                'SOPs_G': sops
            })
            
    with open(csv_path, 'w') as f:
        f.write("Mode,T,FiringRate,SOPs_G\n")
        for r in results_out:
            f.write(f"{r['Mode']},{r['T']},{r['FiringRate']},{r['SOPs_G']}\n")
            
    print(f"\nĐã xuất kết quả SOPs hoàn chỉnh tại: {csv_path}")

if __name__ == '__main__':
    main()
