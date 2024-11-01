#!/usr/bin/env python3
# dlm_attack.py

import argparse
import os
import pickle
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from fedml_api.standalone.superfednas.elastic_nn.ofa_resnets_32x32_10_26 import OFAResNets32x32_10_26

def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def get_supernet():
    model = OFAResNets32x32_10_26().to(DEVICE)
    return model


def load_weight_updates(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")
    with open(pickle_path, 'rb') as f:
        model_paths = pickle.load(f)
    model_updates = [(round_num, model_paths[round_num]) for round_num in model_paths.keys()]
    logging.info(f"Loaded weight updates for {len(model_updates)} rounds from '{pickle_path}'.")
    return model_updates

def load_ground_truth(ground_truth_path):
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found at {ground_truth_path}")
    
    basename = os.path.basename(ground_truth_path)
    try:
        label_str = basename.split('_')[-1]  # 'label3.png'
        label = int(label_str.replace('label', '').replace('.png', ''))
    except (IndexError, ValueError) as e:
        raise ValueError(f"Filename '{basename}' does not match the expected format 'batchX_imgY_labelZ.png'") from e
    
    # Load image
    from PIL import Image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(ground_truth_path).convert('RGB')
    ground_truth_data = transform(image).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, H, W]
    ground_truth_label = torch.tensor([label], dtype=torch.long).to(DEVICE)
    
    logging.info(f"Loaded ground truth image '{ground_truth_path}' with label {label}.")
    return ground_truth_data, ground_truth_label

def visualize_recovered_data(recovered_image, ground_truth_image, save_path=None):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    
    recovered_np = np.transpose(recovered_np, (1, 2, 0))
    ground_truth_np = np.transpose(ground_truth_np, (1, 2, 0))
    
    # Clip values to [0,1] for display
    recovered_np = np.clip(recovered_np, 0, 1)
    ground_truth_np = np.clip(ground_truth_np, 0, 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ground_truth_np)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(recovered_np)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Visualization saved to '{save_path}'.")
    else:
        plt.show()

# ---------------------- Metrics Functions ----------------------

def compute_accuracy(recovered_label, ground_truth_label):
    return int(recovered_label == ground_truth_label)

def compute_psnr(recovered_image, ground_truth_image):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    psnr = compare_psnr(ground_truth_np, recovered_np, data_range=ground_truth_np.max() - ground_truth_np.min())
    return psnr

def compute_ssim(recovered_image, ground_truth_image):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    ssim_score = compare_ssim(recovered_np, ground_truth_np, data_range=ground_truth_np.max() - ground_truth_np.min(), multichannel=True)
    return ssim_score

def compute_lpips_score(recovered_image, ground_truth_image, loss_fn):
    # Ensure the images are in [0,1]
    recovered_image = torch.clamp(recovered_image, 0, 1)
    ground_truth_image = torch.clamp(ground_truth_image, 0, 1)
    
    # Normalize to [-1,1] as required by LPIPS
    recovered_lpips = (recovered_image * 2) - 1 
    ground_truth_lpips = (ground_truth_image * 2) - 1
    
    with torch.no_grad():
        lpips_score = loss_fn(recovered_lpips, ground_truth_lpips).item()
    
    return lpips_score

class DLM_Attack:
    def __init__(self, model, loss_fn, optimizer_class, optimizer_params, target_label, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.target_label = target_label
        self.device = device
        self.reconstructed_data = torch.randn(1, 3, 32, 32, requires_grad=True, device=self.device)
    
    def reconstruct_input(self, target_gradients, num_steps=1000, lr=0.1):
        
        optimizer = self.optimizer_class([self.reconstructed_data], lr=lr)
        
        for step in range(1, num_steps + 1):
            optimizer.zero_grad()
            
            # Forward pass with reconstructed data
            outputs = self.model(self.reconstructed_data)
            loss_ce = self.loss_fn(outputs, self.target_label.view(-1))
            
            # Backward pass to compute gradients
            self.model.zero_grad()
            loss_ce.backward()
            
            # Extract current gradients
            current_gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    current_gradients[name] = param.grad.detach().clone()
            
            # Compute gradient loss (MSE between current and target gradients)
            grad_loss = 0.0
            for name in target_gradients:
                if name in current_gradients:
                    grad_loss += nn.functional.mse_loss(current_gradients[name], target_gradients[name])
                else:
                    logging.warning(f"Gradient '{name}' not found in current gradients.")
            
            # Backward on gradient loss to update reconstructed data
            grad_loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Clamp the reconstructed data to [0,1]
            with torch.no_grad():
                self.reconstructed_data.clamp_(0, 1)
                
            
            # Logging every 100 steps
            if step % 100 == 0 or step == 1:
                logging.info(f"Step [{step}/{num_steps}], Gradient Loss: {grad_loss.item():.6f}")
            
            # Early stopping if gradient loss is sufficiently low
            if grad_loss.item() < 1e-6:
                logging.info(f"[*] Converged at step {step}.")
                break
        
        # After optimization, retrieve the reconstructed data
        recovered_data = self.reconstructed_data.detach().cpu()
        
        # Compute recovered label based on model's prediction
        with torch.no_grad():
            recovered_outputs = self.model(self.reconstructed_data)
            recovered_label = torch.argmax(recovered_outputs, dim=1).item()
        
        logging.info(f"[*] Reconstruction completed. Recovered Label: {recovered_label}")
        
        return recovered_data, recovered_label

def apply_weight_update(model, weight_update):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weight_update:
                param += weight_update[name].to(param.device)
            else:
                logging.warning(f"Parameter '{name}' not found in weight_update.")

def dlm_attack(pickle_path, ground_truth_path, output_dir, num_steps=1000, lr=0.1, device=DEVICE):
    if not os.path.isfile(pickle_path):
        logging.error(f"Pickle file not found: {pickle_path}")
        return

    if not os.path.isfile(ground_truth_path):
        logging.error(f"Ground truth file not found: {ground_truth_path}")
        return


    os.makedirs(output_dir, exist_ok=True)
    

    model = get_supernet()
    logging.info("[*] Initialized supernet model.")

    model.train()

    weight_updates = load_weight_updates(pickle_path)


    ground_truth_data, ground_truth_label = load_ground_truth(ground_truth_path)


    dlm_attacker = DLM_Attack(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_class=optim.SGD,
        optimizer_params={'lr': lr},
        target_label=ground_truth_label,
        device=device
    )


    lpips_fn = lpips.LPIPS(net='alex').to(device)


    for idx, (round_num, update_path) in enumerate(weight_updates):
        logging.info(f"[*] Processing Round {round_num} ({idx+1}/{len(weight_updates)})")

        if not os.path.exists(update_path):
            logging.error(f"Weight update file not found at '{update_path}'. Skipping Round {round_num}.")
            continue


        try:
            with open(update_path, 'rb') as f:
                weight_update = pickle.load(f)
            logging.info(f"Loaded weight update for Round {round_num} from '{update_path}'.")
        except Exception as e:
            logging.error(f"Error loading weight update for Round {round_num}: {e}")
            continue


        try:
            apply_weight_update(model, weight_update)
            logging.info(f"Applied weight update for Round {round_num}.")
        except Exception as e:
            logging.error(f"Error applying weight update for Round {round_num}: {e}")
            continue  


        model.zero_grad()
        outputs = model(ground_truth_data)
        loss = nn.CrossEntropyLoss()(outputs, ground_truth_label.view(-1))
        loss.backward()


        target_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                target_gradients[name] = param.grad.detach().clone()


        recovered_data, recovered_label = dlm_attacker.reconstruct_input(
            target_gradients=target_gradients,
            num_steps=num_steps,
            lr=lr
        )

        # Compute Metrics
        accuracy = compute_accuracy(recovered_label, ground_truth_label.item())
        psnr = compute_psnr(recovered_data, ground_truth_data)
        ssim_score = compute_ssim(recovered_data, ground_truth_data)
        lpips_score = compute_lpips_score(recovered_data, ground_truth_data, lpips_fn)

        logging.info(f"Metrics for Round {round_num}:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"PSNR: {psnr:.2f} dB")
        logging.info(f"SSIM: {ssim_score:.4f}")
        logging.info(f"LPIPS: {lpips_score:.4f}")


        output_path = os.path.join(output_dir, f'recovered_round_{round_num}.pt')

        torch.save({
            'data': recovered_data,
            'label': recovered_label,
            'metrics': {
                'Accuracy': accuracy,
                'PSNR': psnr,
                'SSIM': ssim_score,
                'LPIPS': lpips_score
            }
        }, output_path)
        logging.info(f"[*] Recovered data and metrics saved to '{output_path}'.")

        # Visualize Recovered Data
        visualization_path = os.path.join(output_dir, f'visual_round_{round_num}.png')
        visualize_recovered_data(recovered_data, ground_truth_data, save_path=visualization_path)

    logging.info("[*] DLM attack completed successfully.")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='DLM Attack on SuperFedNAS with CIFAR-10')
    
    parser.add_argument('--pickle', type=str, default='./fed_avg/weight_updates/model_paths.pkl',
                        help='Path to weight updates pickle file')
    parser.add_argument('--ground_truth', type=str, default='./ground_truths/batch7_img9_label3.png',
                        help='Path to ground truth image file')
    parser.add_argument('--output_dir', type=str, default='./recovered_data/',
                        help='Directory to save recovered data and metrics')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for optimizer')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (0-7)')

    args = parser.parse_args()
    
    global DEVICE
    DEVICE = set_device(args.gpu_id)
    logging.info(f"Using device: {DEVICE} (GPU ID: {args.gpu_id})")
    
    dlm_attack(
        pickle_path=args.pickle,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,           
        num_steps=args.steps,
        lr=args.lr,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
