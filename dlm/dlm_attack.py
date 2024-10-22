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


# Define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from fedml_api.standalone.superfednas.Server.ServerModel import ServerResnet_10_26

def get_supernet(n_classes=10, subnet_dist_type='uniform', client_num_in_total=100, bn_gamma_zero_init=False):
    params = {
        "n_classes": n_classes
    }
    model = ServerResnet_10_26(
        params,
        subnet_dist_type,
        client_num_in_total,
        bn_gamma_zero_init,
    ).to(DEVICE)
    return model


def load_weight_updates(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")
    with open(pickle_path, 'rb') as f:
        weight_updates = pickle.load(f)
    logging.info(f"Loaded {len(weight_updates)} weight updates from '{pickle_path}'.")
    return weight_updates

def load_ground_truth(ground_truth_path):
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found at {ground_truth_path}")
    ground_truth = torch.load(ground_truth_path)
    ground_truth_data = ground_truth['data'].to(DEVICE)
    ground_truth_label = ground_truth['label']
    logging.info(f"Loaded ground truth data and label from '{ground_truth_path}'.")
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
    ssim = compare_ssim(ground_truth_np, recovered_np, data_range=ground_truth_np.max() - ground_truth_np.min(), multichannel=True)
    return ssim

def compute_lpips_score(recovered_image, ground_truth_image, loss_fn):
    # LPIPS expects images in [-1,1]
    recovered_lpips = (recovered_image * 2) - 1 
    ground_truth_lpips = (ground_truth_image * 2) - 1
    lpips_score = loss_fn(recovered_lpips, ground_truth_lpips).item()
    return lpips_score



class DLM_Attack:
    def __init__(self, model, loss_fn, optimizer_class, optimizer_params, target_label):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.target_label = target_label
        self.reconstructed_data = torch.randn(1, 3, 32, 32, requires_grad=True, device=DEVICE)
        self.reconstructed_label = nn.Parameter(torch.randn(1, 10, requires_grad=True, device=DEVICE))
    
    def reconstruct_input(self, target_gradients, num_steps=1000, lr=0.1):
        optimizer = self.optimizer_class([self.reconstructed_data, self.reconstructed_label], lr=lr)
        
        # Initialize LPIPS loss function
        lpips_loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
        
        for step in range(1, num_steps + 1):
            optimizer.zero_grad()
            
            # Forward pass with reconstructed data and label logits
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
            
            # Backward on gradient loss to update reconstructed data and label
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
        
        # After optimization, retrieve the reconstructed data and label
        recovered_data = self.reconstructed_data.detach().cpu()
        recovered_label_logits = self.reconstructed_label.detach().cpu()
        recovered_label = torch.argmax(recovered_label_logits, dim=1).item()
        
        logging.info(f"[*] Reconstruction completed. Recovered Label: {recovered_label}")
        
        return recovered_data, recovered_label

def dlm_attack(pickle_path, ground_truth_path, output_path, num_steps=1000, lr=0.1):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load weight updates
    weight_updates = load_weight_updates(pickle_path)
    
    # Load ground truth data and label
    ground_truth_data, ground_truth_label = load_ground_truth(ground_truth_path)
    
    # Initialize the model
    model = get_supernet()
    logging.info("[*] Initialized supernet model.")
    
    # Set model to train mode
    model.train()
    
    # Initialize the DLM attack
    dlm_attacker = DLM_Attack(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': lr},
        target_label=ground_truth_label
    )
    
    # Iterate over each weight update and perform the attack
    for idx, update in enumerate(weight_updates):
        client_id = update['client_id']
        round_num = update['round']
        weight_update = update['weight_update']
        
        logging.info(f"[*] Attacking Weight Update {idx+1}/{len(weight_updates)}: Client {client_id}, Round {round_num}")
        
        # Load the weight update into the model
        model.load_state_dict(weight_update)
        
        # Compute the gradients of the model with respect to the loss using ground truth data
        model.zero_grad()
        outputs = model(ground_truth_data)
        loss = nn.CrossEntropyLoss()(outputs, ground_truth_label.view(-1))
        loss.backward()
        
        # Extract target gradients
        target_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                target_gradients[name] = param.grad.detach().clone()
        
        # Perform the DLM attack to reconstruct the input
        recovered_data, recovered_label = dlm_attacker.reconstruct_input(
            target_gradients=target_gradients,
            num_steps=num_steps,
            lr=lr
        )
        
        
        # Compute Accuracy
        accuracy = compute_accuracy(recovered_label, ground_truth_label.item())
        logging.info(f"Accuracy: {accuracy}")
        
        # Compute PSNR
        psnr = compute_psnr(recovered_data, ground_truth_data)
        logging.info(f"PSNR: {psnr:.2f} dB")
        
        # Compute SSIM
        ssim = compute_ssim(recovered_data, ground_truth_data)
        logging.info(f"SSIM: {ssim:.4f}")
        
        # Compute LPIPS
        lpips_score = compute_lpips_score(recovered_data, ground_truth_data, lpips.LPIPS(net='alex').to(DEVICE))
        logging.info(f"LPIPS: {lpips_score:.4f}")

        metrics = {
            'Accuracy': accuracy,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips_score
        }
        torch.save({
            'data': recovered_data,
            'label': recovered_label,
            'metrics': metrics
        }, output_path)
        logging.info(f"[*] Recovered data and metrics saved to '{output_path}'.")
        

        visualize_recovered_data(recovered_data, ground_truth_data, save_path=f'visual_round_{round_num}_client_{client_id}.png')
    
    logging.info("[*] DLM attack completed successfully.")
    
    return recovered_data, recovered_label, metrics



def main():
    parser = argparse.ArgumentParser(description='DLM Attack on SuperFedNAS with CIFAR-10')
    parser.add_argument('--pickle', type=str, default='all_weight_updates.pkl', help='Path to weight updates pickle file')
    parser.add_argument('--ground_truth', type=str, default='ground_truth.pt', help='Path to ground truth data file')
    parser.add_argument('--output', type=str, default='recovered_data_with_metrics.pt', help='Path to save recovered data and metrics')
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimizer')
    args = parser.parse_args()
    
    recovered_data, recovered_label, metrics = dlm_attack(
        pickle_path=args.pickle,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        num_steps=args.steps,
        lr=args.lr
    )

if __name__ == "__main__":
    main()
