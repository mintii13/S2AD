import torch
import torch.nn as nn
from thop import profile, clever_format
import global_v as glv
import thop
import yaml
from torchvision import models
from svae_models.snn_layers import tdConv, tdConvTranspose, tdBatchNorm, tdLinear
from svae_models.esvae import ESVAE
from svae_models.fsvae import FSVAE
from main_s2ad import build_snn_encoder, BackboneEncoder

# Import SpikingJelly monitor if available
try:
    from spikingjelly.activation_based import monitor
except ImportError:
    monitor = None

def get_flops_and_params(model, input_tensor):
    model.eval()
    custom_ops = {
        tdConv: thop.vision.basic_hooks.count_convNd,
        tdConvTranspose: thop.vision.basic_hooks.count_convNd,
        tdLinear: thop.vision.basic_hooks.count_linear
    }
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_tensor, ), custom_ops=custom_ops, verbose=False)
    flops = macs * 2
    return flops, params

def calculate_snn_energy(model, input_tensor, snn_type="spikingjelly"):
    """
    Measures SOPs by capturing average firing rates of spiking neurons.
    snn_type: "spikingjelly" for S2AD, "custom" for FSVAE/ESVAE
    """
    model.eval()
    
    spike_rates = []
    
    def hook_fn(module, input, output):
        # output is the spike tensor
        # shape usually (T, B, C, H, W) or (B, C, H, W, T) depending on model
        rate = output.float().mean().item()
        spike_rates.append(rate)
        
    hooks = []
    
    # Register hooks based on node type
    for name, module in model.named_modules():
        if snn_type == "spikingjelly":
            import spikingjelly.activation_based.neuron as neuron
            if isinstance(module, neuron.BaseNode):
                hooks.append(module.register_forward_hook(hook_fn))
        else:
            from svae_models.snn_layers import LIFSpike
            # For ESVAE, it's SampledSpikeAct, but we can check if 'Spike' in class name or it's the specific class
            if "Spike" in module.__class__.__name__:
                hooks.append(module.register_forward_hook(hook_fn))
                
    with torch.no_grad():
        _ = model(input_tensor)
        
    for h in hooks:
        h.remove()
        
    avg_firing_rate = sum(spike_rates) / len(spike_rates) if spike_rates else 0.0
    return avg_firing_rate

def report_energy(model_name, flops, params, avg_firing_rate, is_snn=True):
    # Energy parameters (Bu et al. 2022)
    # E_ANN = FLOPs * 12.5 pJ
    # E_SNN = SOPs * 0.9 fJ
    # SOPs = FLOPs * avg_firing_rate
    
    # 1 pJ = 1e-12 J
    # 1 fJ = 1e-15 J
    
    sops = flops * avg_firing_rate if is_snn else 0.0
    
    e_ann_joules = flops * 12.5e-12
    e_snn_joules = sops * 0.9e-15 if is_snn else e_ann_joules
    
    macs_fmt, params_fmt = clever_format([flops/2, params], "%.3f")
    flops_fmt, _ = clever_format([flops, params], "%.3f")
    sops_fmt, _ = clever_format([sops, params], "%.3f") if is_snn else ("N/A", "")
    
    print(f"\n{'='*60}")
    print(f" Profiling Model: {model_name}")
    print(f"{'='*60}")
    print(f"Parameters:        {params_fmt}")
    print(f"FLOPs:             {flops_fmt}")
    if is_snn:
        print(f"Avg Firing Rate:   {avg_firing_rate:.4f}")
        print(f"SOPs:              {sops_fmt}")
        print(f"Energy (ANN ref):  {e_ann_joules*1e3:.4f} mJ")
        print(f"Energy (SNN):      {e_snn_joules*1e6:.4f} uJ")
        print(f"Energy Savings:    {e_ann_joules / e_snn_joules if e_snn_joules > 0 else 0:.2f}x")
    else:
        print(f"Energy (ANN):      {e_ann_joules*1e3:.4f} mJ")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup test batch
    batch_size = 16
    img_size = 256
    
    # --- 1. FSVAE ---
    for T in [8, 16]:
        with open('NetworkConfigs/fsvae_configs/MVTec.yaml', 'r') as f:
            fsvae_config = yaml.safe_load(f)['Network']
        fsvae_config['n_steps'] = T
        glv.init(fsvae_config, ['0'])
        
        fsvae = FSVAE().to(device)
        x_fsvae = torch.randn(batch_size, 3, img_size, img_size, T).to(device)
        
        flops, params = get_flops_and_params(fsvae, x_fsvae)
        rate = calculate_snn_energy(fsvae, x_fsvae, snn_type="custom")
        report_energy(f"FSVAE (T={T}, BS={batch_size})", flops, params, rate, is_snn=True)

    # --- 2. ESVAE ---
    for T in [8, 16]:
        with open('NetworkConfigs/esvae_configs/MVTec.yaml', 'r') as f:
            esvae_config = yaml.safe_load(f)['Network']
        esvae_config['n_steps'] = T
        glv.init(esvae_config, ['0'])
        
        esvae = ESVAE(device, 1.0, 'linear').to(device)
        x_esvae = torch.randn(batch_size, 3, img_size, img_size, T).to(device)
        
        flops, params = get_flops_and_params(esvae, x_esvae)
        rate = calculate_snn_energy(esvae, x_esvae, snn_type="custom")
        report_energy(f"ESVAE (T={T}, BS={batch_size})", flops, params, rate, is_snn=True)

    # --- 3. S2AD Evaluation ---
    s2ad_configs = [
        ('vgg16', '0.4', 16),
        ('alexnet', '0.2', 4),
        ('resnet18', '0.4', 16),
        ('wide_resnet50_2', '0.4', 16)
    ]
    
    for backbone, snn_mode, T in s2ad_configs:
        ann_enc = BackboneEncoder(backbone=backbone, layers='layer123').to(device)
        
        # We compute FLOPs of the ANN backbone for exactly T * batch_size images
        x_ann = torch.randn(T * batch_size, 3, img_size, img_size).to(device)
        flops, params = get_flops_and_params(ann_enc, x_ann)
        
        # Now convert to SNN and measure spike rate
        # S2AD input is (T, B, C, H, W)
        x_snn = torch.randn(T, batch_size, 3, img_size, img_size).to(device)
        
        # Create a dummy calib_loader
        class DummyLoader:
            def __iter__(self):
                yield torch.randn(batch_size, 3, img_size, img_size).to(device), torch.zeros(batch_size)
            def __len__(self):
                return 1
                
        snn_encoder = build_snn_encoder(ann_enc, DummyLoader(), device, snn_mode)
        
        from spikingjelly.activation_based import functional
        functional.reset_net(snn_encoder)
        
        rate = calculate_snn_energy(snn_encoder, x_snn, snn_type="spikingjelly")
        report_energy(f"S2AD {backbone} (Mode={snn_mode}, T={T}, BS={batch_size})", flops, params, rate, is_snn=True)

if __name__ == '__main__':
    main()
