import os
import yaml
import subprocess
import shutil

# Danh sách các class
MVTEC_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

VISA_CLASSES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 
                'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 
                'pcb4', 'pipe_fryum']

def run_save_maps(dataset, classes):
    # Cấu hình theo yêu cầu của bạn
    TARGET_MODE = '0.6'
    TS = 64
    ALPHA = "0.01"
    BATCH_SIZE = 8
    CALIB_SAMPLES = -1 # Full dataset
    
    results_dir = f'./results_anomaly_maps_{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    
    base_config_path = f'NetworkConfigs/s2ad_configs/{"MVTec.yaml" if dataset == "mvtec" else "VisA.yaml"}'
    
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    for cls in classes:
        print(f"\n{'='*60}")
        print(f"[DRAWING MAPS] {dataset.upper()} | Class: {cls}")
        print(f"{'='*60}")
        
        temp_config_path = os.path.join(results_dir, f'temp_{cls}_maps.yaml')
        temp_config = base_config.copy()
        
        # Thiết lập các tham số để lưu ảnh
        temp_config['Network']['snn_mode'] = TARGET_MODE
        temp_config['Network']['timesteps'] = [TS]
        temp_config['Network']['batch_size'] = BATCH_SIZE
        temp_config['Network']['calib_samples'] = CALIB_SAMPLES
        temp_config['Network']['save_anomaly_maps'] = True # BẬT TÍNH NĂNG VẼ ẢNH
        
        # Giữ nguyên cấu hình tốt nhất của S2AD
        temp_config['Network']['layers'] = 'layer123'
        temp_config['Network']['combine_method'] = 'mad_weighted'
        temp_config['Network']['use_zscore'] = True
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)
            
        cmd = [
            "python", "main_s2ad.py",
            "-name", f"maps_run",
            "-category", cls,
            "-config", temp_config_path,
            "-alpha", ALPHA,
            "-project_save_path", results_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"\n[ERROR] Failed to draw maps for {cls}")
            
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    print(f"\n[INFO] Hoàn thành vẽ map cho {dataset.upper()}! Đang đẩy lên Google Drive...")
    source_folder = f"/home/minhtringuyen/ESVAE/results_anomaly_maps_{dataset}/anomaly_maps"
    drive_dest = f"gdrive:/S2AD/Anomalymap_m0.6_a0.01_t64/{dataset}"
    rclone_cmd = f"nohup rclone copy {source_folder} {drive_dest} -v > rclone_{dataset}.log 2>&1 &"
    os.system(rclone_cmd)
    print(f"[RCLONE] Đã gọi ngầm lệnh: {rclone_cmd}")

if __name__ == '__main__':
    print("Bắt đầu tiến trình vẽ Anomaly Maps cho MVTec-AD...")
    run_save_maps('mvtec', MVTEC_CLASSES)
    
    print("\nBắt đầu tiến trình vẽ Anomaly Maps cho VisA...")
    run_save_maps('visa', VISA_CLASSES)
    
    print("\n[HOÀN THÀNH] Toàn bộ tiến trình (vẽ Map + Upload Drive ngầm) đã được kích hoạt!")
