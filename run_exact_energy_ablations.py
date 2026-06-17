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
    print("⏳ Đang đo đạc Firing Rate và Energy CHÍNH XÁC cho các Timestep: 4, 8, 16, 32, 64...")
    
    T_list = [4, 8, 16, 32, 64]
    
    results_dir = "./results_hardware_footprint"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exact_energy_timesteps.csv")
    progress_path = os.path.join(results_dir, "exact_energy_progress.csv")
    
    # Logic Resume: Nếu bị ngắt giữa chừng thì không cần chạy lại các class đã chạy
    processed = {}
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) == 4:
                    T_val, cls, t_mean, t_elements = parts
                    key = f"{T_val}_{cls}"
                    processed[key] = (float(t_mean), float(t_elements))
                    
    ann_enc = BackboneEncoder(backbone='vgg16', layers='layer123').to(device)
    
    VGG_MACS = 20.044578816
    POST_PARAMS = 0.003702784
    TOTAL_OPS = VGG_MACS + POST_PARAMS
    
    results_out = []
    
    for T in T_list:
        print(f"\n{'='*50}")
        print(f"BẮT ĐẦU TÍNH TOÁN CHO T = {T}")
        print(f"{'='*50}")
        
        global_total_mean = 0.0
        global_total_elements = 0.0
        
        for idx, cls in enumerate(MVTEC_CLASSES):
            key = f"{T}_{cls}"
            if key in processed:
                t_mean, t_elements = processed[key]
                global_total_mean += t_mean
                global_total_elements += t_elements
                print(f"  [{idx+1}/15] [SKIP] Đã có data cho {cls} ở T={T}")
                continue
                
            # Đọc toàn bộ ảnh train của class này
            data_path = '/home/minhtringuyen/ANN2SNN/dataloader/datasets/mvtec'
            train_loader, _ = load_mvtec(data_path, cls, batch_size=8, input_size=256)
            
            # Init SNN với Mode 0.6
            snn_encoder = build_snn_encoder(ann_enc, train_loader, device, mode='0.6')
            
            # Tính stats trên tập Train
            stats = compute_normal_stats(snn_encoder, train_loader, device, T, 'layer123')
            
            # Cộng dồn Mean
            cls_total_mean = 0.0
            cls_total_elements = 0.0
            for name, stat in stats.items():
                mean_val = stat['mean'].mean().item()
                num_elements = stat['mean'].numel()
                cls_total_mean += mean_val * num_elements
                cls_total_elements += num_elements
                
            # Ghi vào file progress để resume
            with open(progress_path, 'a') as f:
                f.write(f"{T},{cls},{cls_total_mean},{cls_total_elements}\n")
                
            global_total_mean += cls_total_mean
            global_total_elements += cls_total_elements
            
            print(f"  [{idx+1}/15] [XONG] {cls} (T={T}) | Rate: {cls_total_mean/cls_total_elements:.6f}")
            
        avg_firing_rate = global_total_mean / global_total_elements
        
        sops = TOTAL_OPS * T * avg_firing_rate
        energy_uj = sops * 1e9 * 77.0 * 1e-9 # 77fJ
        
        print(f"\n✅ HOÀN TẤT T={T} | MVTec Avg Firing Rate: {avg_firing_rate:.6f} | SOPs: {sops:.2f} G | Energy: {energy_uj:.1f} uJ\n")
        
        results_out.append({
            'T': T,
            'FiringRate': avg_firing_rate,
            'SOPs_G': sops,
            'Energy_uJ': energy_uj
        })
        
    print("\n" + "="*60)
    print("BẢNG TỔNG KẾT NĂNG LƯỢNG CHUẨN XÁC 100% - S2AD (VGG16)")
    print("="*60)
    header = f"{'T':<4} | {'Firing Rate':>12} | {'SOPs (G)':>10} | {'Energy (uJ)':>15}"
    print(header)
    print("-" * len(header))
    
    with open(csv_path, 'w') as f:
        f.write("T,FiringRate,SOPs_G,Energy_uJ\n")
        for r in results_out:
            print(f"{r['T']:<4} | {r['FiringRate']:>12.6f} | {r['SOPs_G']:>10.2f} | {r['Energy_uJ']:>15.1f}")
            f.write(f"{r['T']},{r['FiringRate']},{r['SOPs_G']},{r['Energy_uJ']}\n")
            
    print(f"\nĐã lưu kết quả hoàn chỉnh vào: {csv_path}")
    print("Bạn có thể copy chính xác các con số này vào Table 9 và Table 10.")

if __name__ == '__main__':
    main()
