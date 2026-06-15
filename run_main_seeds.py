import os
import yaml
import subprocess
import pandas as pd

MVTEC_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
VISA_CLASSES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

def run_all_timesteps_seeds(dataset, classes, mode, timesteps_list, alpha, seeds):
    base_config_path = f'NetworkConfigs/s2ad_configs/{"MVTec.yaml" if dataset == "mvtec" else "VisA.yaml"}'
    results_dir = f'./results_main_seeds_{dataset}'
    csv_path = os.path.join(results_dir, f'paper_main_{dataset}_seeds.csv')
    os.makedirs(results_dir, exist_ok=True)
    
    # Khởi tạo CSV
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("Class,Seed,Mode,Timesteps,Alpha,ImgAUC,ImgAP,ImgF1,PixAUC,PixAP,PixF1,PRO,mAD,MAC,SOP,CalibTime,TestTime,FPS\n")
            
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"[{dataset.upper()}] RUNNING SEED: {seed} | TIMESTEPS: {timesteps_list}")
            print(f"{'='*60}")
            
            for cls in classes:
                df_temp = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["Class", "Seed"])
                # Nếu đã chạy đủ 5 timesteps cho class và seed này thì skip
                if len(df_temp[(df_temp['Class'] == cls) & (df_temp['Seed'] == seed)]) >= len(timesteps_list):
                    print(f"Skipping {cls} Seed={seed} (already done)")
                    continue
                
                temp_config_path = os.path.join(results_dir, f'temp_{cls}_{seed}.yaml')
                temp_config = base_config.copy()
                
                temp_config['Network']['snn_mode'] = mode
                temp_config['Network']['timesteps'] = timesteps_list # CHẠY 1 LẦN FULL 5 TIMESTEPS
                temp_config['Network']['layers'] = 'layer123'
                temp_config['Network']['combine_method'] = 'mad_weighted'
                temp_config['Network']['use_zscore'] = True
                temp_config['Network']['save_anomaly_maps'] = False
                
                with open(temp_config_path, 'w') as f:
                    yaml.dump(temp_config, f)
                    
                cmd = [
                    "python", "main_s2ad.py",
                    "-name", f"{dataset}_main",
                    "-category", cls,
                    "-config", temp_config_path,
                    "-alpha", str(alpha),
                    "-seed", str(seed),
                    "-project_save_path", results_dir
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    
                    # Quét qua từng timestep để đọc đúng file log
                    for t_val in timesteps_list:
                        # Tên file xuất ra từ main_s2ad.py: f'{dataset_name}_{category_name}_T{T}_ad_eval_results.txt'
                        log_file = os.path.join(results_dir, f"{dataset}_{cls}_T{t_val}_ad_eval_results.txt")
                        if os.path.exists(log_file):
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                # Dòng cuối cùng chứa data (dòng 2 là data)
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
                                        
                                        row = f"{cls},{seed},{mode},{t_val},{alpha},{img_auc:.4f},{img_ap:.4f},{img_f1:.4f},{pix_auc:.4f},{pix_ap:.4f},{pix_f1:.4f},{pro:.4f},{mad:.4f},{mac:.2f},{sop:.2f},{calib_time:.2f},{test_time:.2f},{fps:.2f}\n"
                                        with open(csv_path, 'a') as fc:
                                            fc.write(row)
                            os.remove(log_file)
                            
                    # Xoá luôn cả file tổng hợp s2ad_results nếu có
                    s2ad_res = os.path.join(results_dir, f"{cls}_s2ad_results.txt")
                    if os.path.exists(s2ad_res): os.remove(s2ad_res)
                    
                except Exception as e:
                    print(f"[ERROR] failed {cls} at seed {seed}")
                    
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

    # -------------------------------------------------------------
    # Import Seed 42 from Grid Search
    # -------------------------------------------------------------
    grid_csv_path = f'/home/minhtringuyen/ESVAE/results_grid_mains2ad_{dataset}/paper_grid_{dataset}_mains2ad.csv'
    if os.path.exists(grid_csv_path):
        df_grid = pd.read_csv(grid_csv_path)
        # Import Seed 42 cho tất cả timesteps
        df_seed42 = df_grid[(df_grid['Mode'] == mode) & (df_grid['Timesteps'].isin(timesteps_list)) & (df_grid['Alpha'] == alpha)].copy()
        if not df_seed42.empty:
            df_seed42['Seed'] = 42
            current_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["Seed"])
            
            for t in timesteps_list:
                df_t42 = df_seed42[df_seed42['Timesteps'] == t]
                if not df_t42.empty:
                    # Rename columns if needed
                    df_t42 = df_t42.rename(columns={'MAC(G)':'MAC', 'SOP(G)':'SOP', 'CalibTime(s)':'CalibTime', 'TestTime(s)':'TestTime'})
                    df_t42 = df_t42[['Class','Seed','Mode','Timesteps','Alpha','ImgAUC','ImgAP','ImgF1','PixAUC','PixAP','PixF1','PRO','mAD','MAC','SOP','CalibTime','TestTime','FPS']]
                    
                    # Nếu chưa có trong csv thì add
                    if current_df.empty or current_df[(current_df['Seed'] == 42) & (current_df['Timesteps'] == t)].empty:
                        df_t42.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    # -------------------------------------------------------------
    # Tổng hợp tính Mean và Std cho TỪNG TIMESTEP
    # -------------------------------------------------------------
    print(f"\n[SUMMARY] {dataset.upper()} - Tính Mean ± Std cho TỪNG Timestep")
    df = pd.read_csv(csv_path)
    
    summary_file = os.path.join(results_dir, f'final_results_all_timesteps_{dataset}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"=== {dataset.upper()} COMPREHENSIVE CONFIG: Mode={mode}, Alpha={alpha} ===\n")
        f.write(f"Seeds evaluated: {df['Seed'].unique().tolist()}\n")
        f.write("Lưu ý: Các metric hiệu năng hiển thị dưới dạng Mean ± Std (%). Các thông số thời gian là Mean.\n\n")
        
        # In Header
        metrics_header = f"{'T':>3} | {'ImgAUC':>14} | {'ImgAP':>14} | {'ImgF1':>14} | {'PixAUC':>14} | {'PixAP':>14} | {'PixF1':>14} | {'PRO':>14} | {'mAD':>14} | {'MAC':>6} | {'SOP':>6} | {'Calib':>6} | {'Test':>6} | {'FPS':>6}"
        f.write(metrics_header + "\n")
        f.write("-" * 190 + "\n")
        
        for t in timesteps_list:
            df_t = df[df['Timesteps'] == t]
            if df_t.empty: continue
            
            # Tính trung bình điểm của các class ĐỐI VỚI TỪNG SEED
            seed_scores = df_t.groupby('Seed').mean(numeric_only=True)
            
            # Sau đó tính Mean và Std GIỮA CÁC SEED
            mean_val = seed_scores.mean()
            std_val = seed_scores.std()
            
            # Nếu chỉ có 1 seed thì std = 0
            std_val = std_val.fillna(0.0)
            
            row_str = (f"{t:>3} | "
                       f"{mean_val['ImgAUC']*100:>6.2f} ± {std_val['ImgAUC']*100:>4.2f} | "
                       f"{mean_val['ImgAP']*100:>6.2f} ± {std_val['ImgAP']*100:>4.2f} | "
                       f"{mean_val['ImgF1']*100:>6.2f} ± {std_val['ImgF1']*100:>4.2f} | "
                       f"{mean_val['PixAUC']*100:>6.2f} ± {std_val['PixAUC']*100:>4.2f} | "
                       f"{mean_val['PixAP']*100:>6.2f} ± {std_val['PixAP']*100:>4.2f} | "
                       f"{mean_val['PixF1']*100:>6.2f} ± {std_val['PixF1']*100:>4.2f} | "
                       f"{mean_val['PRO']*100:>6.2f} ± {std_val['PRO']*100:>4.2f} | "
                       f"{mean_val['mAD']*100:>6.2f} ± {std_val['mAD']*100:>4.2f} | "
                       f"{mean_val['MAC']:>6.2f} | {mean_val['SOP']:>6.2f} | "
                       f"{mean_val['CalibTime']:>6.1f} | {mean_val['TestTime']:>6.2f} | {mean_val['FPS']:>6.1f}\n")
            f.write(row_str)
            
    print(f"Done! Kết quả lưu tại: {summary_file}")

if __name__ == '__main__':
    seeds_to_run = [0, 2026]
    timesteps_list = [4, 8, 16, 32, 64]
    
    print("\n[1] Bắt đầu chạy MVTec-AD...")
    run_all_timesteps_seeds('mvtec', MVTEC_CLASSES, mode='0.6', timesteps_list=timesteps_list, alpha=0.01, seeds=seeds_to_run)
    
    print("\n[2] Bắt đầu chạy VisA...")
    run_all_timesteps_seeds('visa', VISA_CLASSES, mode='0.6', timesteps_list=timesteps_list, alpha=0.01, seeds=seeds_to_run)
