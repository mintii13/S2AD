import os

CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]
RESULTS_DIR = './results_esvae_d1024'
t = 16

def get_last_metrics(dataset_name, category, n_steps):
    file_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{category}_T{n_steps}_ad_eval_results.txt")
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
                    'mad': float(parts[8]),
                    'train_loss': float(parts[9]),
                    'test_loss': float(parts[10]),
                    'train_time': float(parts[11]),
                    'test_time': float(parts[12]),
                    'fps': float(parts[13])
                }
            elif len(parts) >= 11:
                return {
                    'img_auc': float(parts[1]),
                    'img_ap': float(parts[2]),
                    'img_f1': float(parts[3]),
                    'pix_auc': float(parts[4]),
                    'pix_ap': float(parts[5]),
                    'pix_f1': float(parts[6]),
                    'pro': float(parts[7]),
                    'mad': float(parts[8]),
                    'train_loss': 0.0,
                    'test_loss': 0.0,
                    'train_time': 0.0,
                    'test_time': float(parts[9]),
                    'fps': float(parts[10])
                }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

summary_file = os.path.join(RESULTS_DIR, f'mvtec_overall_summary_T{t}.txt')
all_metrics = {}

with open(summary_file, 'w') as f:
    f.write(f"=== MVTec Overall Summary [Timestep {t}] ===\n")
    f.write(f"{'Class':<15} | {'Img AUC':>7} | {'Img AP':>7} | {'Img F1':>7} | {'Pix AUC':>7} | {'Pix AP':>7} | {'Pix F1':>7} | {'PRO':>7} | {'mAD':>7} | {'TrainLoss':>9} | {'TestLoss':>9} | {'Train(s)':>8} | {'Test(s)':>7} | {'FPS':>7}\n")
    f.write("-" * 145 + "\n")

    for cls in CLASSES:
        metrics = get_last_metrics('mvtec', cls, t)
        if metrics is not None:
            all_metrics[cls] = metrics
            m = metrics
            f.write(f"{cls:<15} | {m['img_auc']:7.4f} | {m['img_ap']:7.4f} | {m['img_f1']:7.4f} | {m['pix_auc']:7.4f} | {m['pix_ap']:7.4f} | {m['pix_f1']:7.4f} | {m['pro']:7.4f} | {m['mad']:7.4f} | {m['train_loss']:9.4f} | {m['test_loss']:9.4f} | {m['train_time']:8.1f} | {m['test_time']:7.1f} | {m['fps']:7.1f}\n")
        else:
            print(f"Missing {cls}")

    if all_metrics:
        f.write("=" * 145 + "\n")
        avg_metrics = {}
        for key in ['img_auc', 'img_ap', 'img_f1', 'pix_auc', 'pix_ap', 'pix_f1', 'pro', 'mad', 'train_loss', 'test_loss', 'train_time', 'test_time', 'fps']:
            vals = [m[key] for m in all_metrics.values()]
            avg_metrics[key] = sum(vals) / len(vals) if vals else 0.0
        am = avg_metrics
        f.write(f"{'AVERAGE':<15} | {am['img_auc']:7.4f} | {am['img_ap']:7.4f} | {am['img_f1']:7.4f} | {am['pix_auc']:7.4f} | {am['pix_ap']:7.4f} | {am['pix_f1']:7.4f} | {am['pro']:7.4f} | {am['mad']:7.4f} | {am['train_loss']:9.4f} | {am['test_loss']:9.4f} | {am['train_time']:8.1f} | {am['test_time']:7.1f} | {am['fps']:7.1f}\n")

print("Done")
