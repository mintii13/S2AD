import os
import yaml
import subprocess
import time
import argparse
import csv

CLASSES_MVTEC = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

CLASSES_VISA = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

def update_yaml_ablation(path, combine_method, use_zscore, n_steps, backbone, snn_mode):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    data['Network']['combine_method'] = combine_method
    data['Network']['use_zscore'] = use_zscore
    data['Network']['timesteps'] = [n_steps]
    data['Network']['n_steps'] = n_steps
    data['Network']['backbone'] = backbone
    data['Network']['snn_mode'] = str(snn_mode)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def get_last_metrics(dataset_name, category, n_steps, results_dir):
    file_path = os.path.join(results_dir, f"{dataset_name}_{category}_T{n_steps}_ad_eval_results.txt")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('-') and not line.startswith('Epoch')]
            if not lines:
                return None
            last_line = lines[-1]
            parts = [p.strip() for p in last_line.split('|')]
            if len(parts) >= 14:
                return {
                    'img_auc': float(parts[1]),
                    'img_ap': float(parts[2]),
                    'img_f1': float(parts[3]),
                    'pix_auc': float(parts[4]),
                    'pix_ap': float(parts[5]),
                    'pix_f1': float(parts[6]),
                    'pro': float(parts[7]),
                    'mad': float(parts[8])
                }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, default=16)
    parser.add_argument('-backbone', type=str, default='vgg16')
    parser.add_argument('-snn_mode', type=str, default='0.4')
    args = parser.parse_args()

    datasets = ['mvtec', 'visa']
    
    # Ablation settings: (combine_method, use_zscore, label_mad, label_zscore)
    # Skipped full setting ('mad_weighted', True, 'X', 'X') as requested.
    ablations = [
        ('simple', False, '', ''),
        ('mad_weighted', False, 'X', ''),
        ('simple', True, '', 'X')
    ]
    
    for dataset in datasets:
        classes = CLASSES_MVTEC if dataset == 'mvtec' else CLASSES_VISA
        config_path = f'NetworkConfigs/s2ad_configs/{"MVTec" if dataset == "mvtec" else "VisA"}.yaml'
        output_csv = f'ablation_results_{dataset}.csv'
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['MAD weighting', 'Z-score', 'Image AU-ROC', 'Image AP', 'Image F1', 'Pixel AU-ROC', 'Pixel AP', 'Pixel F1', 'AU-PRO', 'mAD'])
        
        print(f"\n{'='*60}")
        print(f" STARTING ABLATION STUDY FOR {dataset.upper()}...")
        print(f"{'='*60}")
        
        for combine_method, use_zscore, label_mad, label_zscore in ablations:
            run_name = f"{'MAD' if label_mad else 'NoMAD'}_{'ZScore' if label_zscore else 'NoZScore'}"
            results_dir = f'./results_ablation_{dataset}/{run_name}'
            os.makedirs(results_dir, exist_ok=True)
            
            summary_file = os.path.join(results_dir, f'{dataset}_overall_summary_T{args.t}.txt')
            with open(summary_file, 'w') as f:
                f.write(f"=== {dataset.upper()} Overall Summary [T={args.t} | MAD={label_mad=='X'} | Z-Score={label_zscore=='X'}] ===\n")
                f.write(f"{'Class':<15} | {'Img AUC':>7} | {'Img AP':>7} | {'Img F1':>7} | {'Pix AUC':>7} | {'Pix AP':>7} | {'Pix F1':>7} | {'PRO':>7} | {'mAD':>7}\n")
                f.write("-" * 100 + "\n")
            
            print(f"\n{'='*60}")
            print(f" ABLATION RUN [{dataset.upper()}]: MAD weighting = {label_mad=='X'}, Z-score = {label_zscore=='X'} (T={args.t}, {args.backbone})")
            print(f"{'='*60}")
            
            update_yaml_ablation(config_path, combine_method, use_zscore, args.t, args.backbone, args.snn_mode)
            
            all_metrics = {}
            for cls in classes:
                cmd = [
                    "python", "main_s2ad.py",
                    "-name", f"ablation_{run_name}",
                    "-category", cls,
                    "-config", config_path,
                    "-project_save_path", results_dir
                ]
                
                existing = get_last_metrics(dataset, cls, args.t, results_dir)
                if not existing:
                    subprocess.run(cmd)
                    existing = get_last_metrics(dataset, cls, args.t, results_dir)
                
                if existing:
                    all_metrics[cls] = existing
                    with open(summary_file, 'a') as f:
                        m = existing
                        f.write(f"{cls:<15} | {m['img_auc']:7.4f} | {m['img_ap']:7.4f} | {m['img_f1']:7.4f} | {m['pix_auc']:7.4f} | {m['pix_ap']:7.4f} | {m['pix_f1']:7.4f} | {m['pro']:7.4f} | {m['mad']:7.4f}\n")
                    
            if not all_metrics:
                continue
                
            # Calculate averages
            avg_metrics = {}
            for key in ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_ap', 'pix_f1', 'pro', 'mad']:
                vals = [m[key] for m in all_metrics.values()]
                avg_metrics[key] = sum(vals) / len(vals) if vals else 0.0
                
            with open(summary_file, 'a') as f:
                f.write("=" * 100 + "\n")
                am = avg_metrics
                f.write(f"{'AVERAGE':<15} | {am['img_auc']:7.4f} | {am['img_ap']:7.4f} | {am['img_f1']:7.4f} | {am['pix_auc']:7.4f} | {am['pix_ap']:7.4f} | {am['pix_f1']:7.4f} | {am['pro']:7.4f} | {am['mad']:7.4f}\n")
                
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    label_mad, 
                    label_zscore, 
                    f"{avg_metrics['img_auc']:.4f}",
                    f"{avg_metrics['img_ap']:.4f}",
                    f"{avg_metrics['img_f1']:.4f}",
                    f"{avg_metrics['pix_auc']:.4f}",
                    f"{avg_metrics['pix_ap']:.4f}",
                    f"{avg_metrics['pix_f1']:.4f}",
                    f"{avg_metrics['pro']:.4f}",
                    f"{avg_metrics['mad']:.4f}"
                ])
                
        print(f"\n[DONE] Ablation results for {dataset.upper()} saved to {output_csv}")

if __name__ == '__main__':
    main()
