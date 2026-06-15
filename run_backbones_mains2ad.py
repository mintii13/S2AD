import os
import yaml
import subprocess
import pandas as pd
import copy

MVTEC_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
VISA_CLASSES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

# Danh sách 6 backbone (đã bỏ resnet34, đẩy alexnet lên đầu)
BACKBONES = ['alexnet', 'resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11']

def run_backbones(dataset, classes, mode, timesteps_list, alpha, backbones_to_run):
    base_config_path = f'NetworkConfigs/s2ad_configs/{"MVTec.yaml" if dataset == "mvtec" else "VisA.yaml"}'
    results_dir = f'./results_backbone_mains2ad_{dataset}'
    csv_path = os.path.join(results_dir, f'paper_backbone_{dataset}.csv')
    os.makedirs(results_dir, exist_ok=True)
    
    # Khởi tạo CSV
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("Class,Backbone,Mode,Timesteps,Alpha,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,MAC,SOP,CalibTime,TestTime,FPS\n")
            
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    for backbone in backbones_to_run:
        print(f"\n{'='*60}")
        print(f"[{dataset.upper()}] RUNNING BACKBONE: {backbone} | TIMESTEPS: {timesteps_list}")
        print(f"{'='*60}")
        
        for cls in classes:
            df_temp = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["Class", "Backbone"])
            if len(df_temp[(df_temp['Class'] == cls) & (df_temp['Backbone'] == backbone)]) >= len(timesteps_list):
                print(f"Skipping {cls} Backbone={backbone} (already done)")
                continue
            
            temp_config_path = os.path.join(results_dir, f'temp_{cls}_{backbone}.yaml')
            temp_config = copy.deepcopy(base_config)
            
            temp_config['Network']['backbone'] = backbone
            temp_config['Network']['layers'] = 'layer123'
                
            temp_config['Network']['snn_mode'] = mode
            temp_config['Network']['timesteps'] = timesteps_list
            temp_config['Network']['combine_method'] = 'mad_weighted'
            temp_config['Network']['use_zscore'] = True
            temp_config['Network']['save_anomaly_maps'] = False
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
                
            cmd = [
                "python", "main_s2ad.py",
                "-name", f"{dataset}_backbone",
                "-category", cls,
                "-config", temp_config_path,
                "-alpha", str(alpha),
                "-seed", "42",  # Cố định seed 42 cho bảng Ablation
                "-project_save_path", results_dir
            ]
            
            try:
                subprocess.run(cmd, check=True)
                
                for t_val in timesteps_list:
                    log_file = os.path.join(results_dir, f"{dataset}_{cls}_T{t_val}_ad_eval_results.txt")
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 2:
                                parts = [p.strip() for p in lines[-1].split('|')]
                                if len(parts) >= 14:
                                    img_auc, img_ap, img_f1 = float(parts[1]), float(parts[2]), float(parts[3])
                                    pix_auc, pix_ap, pix_f1 = float(parts[4]), float(parts[5]), float(parts[6])
                                    pro, mad = float(parts[7]), float(parts[8])
                                    mac, sop = 0.0, 0.0
                                    calib_time = float(parts[11])
                                    test_time = float(parts[12])
                                    fps = float(parts[13])
                                    
                                    row = f"{cls},{backbone},{mode},{t_val},{alpha},{img_auc:.4f},{img_ap:.4f},{img_f1:.4f},{pix_auc:.4f},{pix_ap:.4f},{pix_f1:.4f},{pro:.4f},{mad:.4f},{mac:.2f},{sop:.2f},{calib_time:.2f},{test_time:.2f},{fps:.2f}\n"
                                    with open(csv_path, 'a') as fc:
                                        fc.write(row)
                        os.remove(log_file)
                        
                s2ad_res = os.path.join(results_dir, f"{cls}_s2ad_results.txt")
                if os.path.exists(s2ad_res): os.remove(s2ad_res)
                
            except Exception as e:
                print(f"[ERROR] failed {cls} at backbone {backbone}")
                
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    # -------------------------------------------------------------
    # TỔNG HỢP SUMMARY BẢNG BACKBONE (LẤY TRUNG BÌNH THEO TIMESTEP)
    # -------------------------------------------------------------
    print(f"\n[SUMMARY] Đang tạo file text tổng hợp cho {dataset.upper()}...")
    df = pd.read_csv(csv_path)
    
    summary_file = os.path.join(results_dir, f'final_backbone_summary_{dataset}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"=== {dataset.upper()} BACKBONE ABLATION: Mode={mode}, Alpha={alpha} ===\n\n")
        
        # Nhóm theo Backbone và Timesteps, tính trung bình của các classes
        avg_df = df.groupby(['Backbone', 'Timesteps']).mean(numeric_only=True).reset_index()
        
        header = f"{'Backbone':<16} | {'T':>3} | {'ImgAUC':>8} | {'ImgAP':>8} | {'ImgF1':>8} | {'PixAUC':>8} | {'PixAP':>8} | {'PixF1':>8} | {'PRO':>8} | {'mAD':>8} | {'Calib':>6} | {'Test':>6} | {'FPS':>6}"
        f.write(header + "\n")
        f.write("-" * 150 + "\n")
        
        for bb in BACKBONES:
            df_bb = avg_df[avg_df['Backbone'] == bb]
            if df_bb.empty: continue
            
            for _, row in df_bb.iterrows():
                f.write(f"{row['Backbone']:<16} | {int(row['Timesteps']):>3} | "
                        f"{row['ImgAUC']*100:>8.2f} | {row['ImgAP']*100:>8.2f} | {row['ImgF1']*100:>8.2f} | "
                        f"{row['PixAUC']*100:>8.2f} | {row['PixAP']*100:>8.2f} | {row['PixF1']*100:>8.2f} | "
                        f"{row['PRO']*100:>8.2f} | {row['mAD']*100:>8.2f} | "
                        f"{row['CalibTime']:>6.1f} | {row['TestTime']:>6.2f} | {row['FPS']:>6.1f}\n")
            f.write("-" * 150 + "\n")
            
    print(f"Done! Kết quả lưu tại: {summary_file}")

if __name__ == '__main__':
    timesteps_list = [4, 8, 16, 32, 64]
    mode_target = '0.6'
    alpha_target = 0.01
    
    print("\nBẮT ĐẦU QUÉT BACKBONE (MVTec -> VisA)")
    for bb in BACKBONES:
        run_backbones('mvtec', MVTEC_CLASSES, mode_target, timesteps_list, alpha_target, [bb])
        run_backbones('visa', VISA_CLASSES, mode_target, timesteps_list, alpha_target, [bb])
