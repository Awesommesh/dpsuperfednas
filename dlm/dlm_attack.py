#!/usr/bin/env python3
# dlm_attack_refactored.py

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

from PIL import Image

from fedml_api.standalone.superfednas.elastic_nn.ofa_resnets_32x32_10_26 import OFAResNets32x32_10_26


def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_weight_updates(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        model_paths = pickle.load(f)
    
    model_updates = []
    for round_num in sorted(model_paths.keys()):
        update_path = model_paths[round_num]
        model_updates.append((round_num, update_path))
    
    logging.info(f"Loaded weight updates for {len(model_updates)} rounds from '{pickle_path}'.")
    return model_updates


def load_ground_truth_images(ground_truth_dir, target_label=None):
    if not os.path.exists(ground_truth_dir):
        raise FileNotFoundError(f"Ground truth directory not found at {ground_truth_dir}")
    
    label_to_image = {}
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    for filename in os.listdir(ground_truth_dir):
        if not filename.endswith('.png'):
            continue 
        
        filepath = os.path.join(ground_truth_dir, filename)
        

        try:
            label_str = filename.split('_')[-1]  # e.g., 'label3.png'
            label = int(label_str.replace('label', '').replace('.png', ''))
        except (IndexError, ValueError) as e:
            logging.warning(f"Filename '{filename}' does not match the expected format. Skipping.")
            continue
        
        # If a target_label is specified and the current label does not match, skip
        if target_label is not None and label != target_label:
            continue
        
        # If we've already loaded an image for this label, skip
        if label in label_to_image:
            continue
        
        # Load and transform the image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, H, W]
        
        label_to_image[label] = image_tensor
        logging.info(f"Loaded ground truth image '{filename}' with label {label}.")
        
        # If focusing on a specific label and we've loaded it, break
        if target_label is not None and label in label_to_image:
            break
    
    if not label_to_image:
        raise ValueError(f"No ground truth images found in '{ground_truth_dir}' for the specified criteria.")
    
    return label_to_image


def compute_accuracy(recovered_label, ground_truth_label):
    return int(recovered_label == ground_truth_label)


def compute_psnr_metric(recovered_image, ground_truth_image):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    psnr = compare_psnr(ground_truth_np, recovered_np, data_range=ground_truth_np.max() - ground_truth_np.min())
    return psnr


def compute_ssim_metric(recovered_image, ground_truth_image):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    
    # Debugging information
    logging.debug(f"Recovered image shape: {recovered_np.shape}")
    logging.debug(f"Ground truth image shape: {ground_truth_np.shape}")
    
    # Check if images have 3 channels; if so, transpose to [H, W, C]
    if recovered_np.ndim == 3 and recovered_np.shape[0] == 3:
        recovered_np = np.transpose(recovered_np, (1, 2, 0))  # [H, W, C]
    if ground_truth_np.ndim == 3 and ground_truth_np.shape[0] == 3:
        ground_truth_np = np.transpose(ground_truth_np, (1, 2, 0))  # [H, W, C]
    
    if recovered_np.ndim == 3:
        height, width, channels = recovered_np.shape
    elif recovered_np.ndim == 2:
        height, width = recovered_np.shape
        channels = None
    else:
        raise ValueError("recovered_np and ground_truth_np must be 2D or 3D arrays")
    
    # Determine appropriate win_size
    win_size = 7  # Default value
    if height < win_size or width < win_size:
        win_size = min(height, width)
        if win_size % 2 == 0:
            win_size -= 1  
        if win_size < 3:
            logging.error("Images are too small for SSIM computation with win_size < 3.")
            raise ValueError("Images are too small for SSIM computation.")
        logging.warning(f"Adjusted win_size to {win_size} due to small image dimensions.")
    
    try:
        ssim_score = compare_ssim(
            recovered_np,
            ground_truth_np,
            data_range=ground_truth_np.max() - ground_truth_np.min(),
            channel_axis=2 if channels == 3 else None,
            win_size=win_size
        )
    except ValueError as e:
        logging.error(f"SSIM computation failed: {e}")
        raise
    
    return ssim_score


def compute_lpips_score(recovered_image, ground_truth_image, loss_fn):
    try:
        device = next(loss_fn.parameters()).device
    except StopIteration:
        logging.error("LPIPS model has no parameters to determine device.")
        raise RuntimeError("LPIPS model has no parameters to determine device.")
    
    # Ensure the images are in [0,1] and moved to the LPIPS model's device
    recovered_image = torch.clamp(recovered_image, 0, 1).to(device)
    ground_truth_image = torch.clamp(ground_truth_image, 0, 1).to(device)
    
    # Normalize to [-1,1] as required by LPIPS
    recovered_lpips = (recovered_image * 2) - 1 
    ground_truth_lpips = (ground_truth_image * 2) - 1
    
    with torch.no_grad():
        lpips_score = loss_fn(recovered_lpips, ground_truth_lpips).item()
    
    return lpips_score


def find_closest_image(recovered_image, ground_truth_dict, metric_fn, same_class=False, target_label=None):
    best_score = -float('inf')  # Initialize to negative infinity for maximization metrics like PSNR
    closest_label = None
    closest_image = None
    
    for label, gt_image in ground_truth_dict.items():
        if same_class:
            if target_label is None:
                raise ValueError("target_label must be specified when same_class is True.")
            if label != target_label:
                continue  # Skip images not in the target class
        
        score = metric_fn(recovered_image, gt_image)
        
        if np.isnan(score):
            logging.warning(f"Metric function returned NaN for label {label}. Skipping.")
            continue
        
        if score > best_score:
            best_score = score
            closest_label = label
            closest_image = gt_image
    
    if closest_image is None:
        logging.warning(f"No valid closest image found for label {target_label}.")
    
    return closest_label, closest_image, best_score


def find_closest_image_same_class(recovered_image, ground_truth_dict, metric_fn, target_label):
    return find_closest_image(
        recovered_image,
        ground_truth_dict,
        metric_fn,
        same_class=True,
        target_label=target_label
    )


def visualize_recovered_data(recovered_image, ground_truth_image, closest_image_same_class, save_path=None):

    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    
    # Initialize closest_same_np to None
    closest_same_np = None
    
    if closest_image_same_class is not None:
        closest_same_np = closest_image_same_class.squeeze().detach().cpu().numpy()
    
        # Transpose if necessary
        if recovered_np.ndim == 3 and recovered_np.shape[0] == 3:
            recovered_np = np.transpose(recovered_np, (1, 2, 0))  # [H, W, C]
        if ground_truth_np.ndim == 3 and ground_truth_np.shape[0] == 3:
            ground_truth_np = np.transpose(ground_truth_np, (1, 2, 0))  # [H, W, C]
        if closest_same_np.ndim == 3 and closest_same_np.shape[0] == 3:
            closest_same_np = np.transpose(closest_same_np, (1, 2, 0))  # [H, W, C]
        
        # Clip values to [0,1] for display
        closest_same_np = np.clip(closest_same_np, 0, 1)
    
    # Clip values to [0,1] for display
    recovered_np = np.clip(recovered_np, 0, 1)
    ground_truth_np = np.clip(ground_truth_np, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(ground_truth_np)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(recovered_np)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    if closest_same_np is not None:
        axes[2].imshow(closest_same_np)
        axes[2].set_title('Closest Same Class')
    else:
        # If no closest image is found, display a blank image or a message
        blank_image = np.ones_like(recovered_np)
        axes[2].imshow(blank_image)
        axes[2].set_title('No Closest Image Found')
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Visualization saved to '{save_path}'.")
    else:
        plt.show()


def plot_metrics(loss_history, psnr_history, output_dir, round_num, target_label):
    plt.figure(figsize=(12, 5))

    # Loss History
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Total Loss')
    plt.title(f'Round {round_num} - Label {target_label} - Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # PSNR History
    plt.subplot(1, 2, 2)
    plt.plot(psnr_history, label='PSNR', color='orange')
    plt.title(f'Round {round_num} - Label {target_label} - PSNR History')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'metrics_round_{round_num}_label_{target_label}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"[*] Metrics plot saved to '{plot_path}'.")


def get_supernet():
    model = OFAResNets32x32_10_26().to(DEVICE)
    model.set_active_subnet(d=0, e=0.1, w=0)  
    model = model.get_active_subnet()
    
    # Ensure all parameters require gradients
    for name, param in model.named_parameters():
        param.requires_grad = True
        if param.grad is not None:
            param.grad.zero_()
    
    return model




class DLM_Attack:
    """
    Performs the DLM attack to reconstruct input data by matching gradients.
    """
    def __init__(self, model, loss_fn, optimizer_class, optimizer_params, target_label, ground_truth_data, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.target_label = target_label
        self.device = device
        # Initialize reconstructed data with the same shape and type as ground truth data
        self.reconstructed_data = torch.randn_like(ground_truth_data).float().to(self.device).requires_grad_(True)
    
    def reconstruct_input(self, normalized_target_grads, ground_truth_data, num_steps=1000, lr=0.1):
        optimizer = self.optimizer_class([self.reconstructed_data], **self.optimizer_params)
    
        loss_history = []
        psnr_history = []
        total_loss_list = []  # Mutable container to capture total_loss
    
        offset = 1e-8  # Small constant to prevent division by zero
    
        for step in range(1, num_steps + 1):
            def closure():
                optimizer.zero_grad()
    
                # Forward pass with reconstructed data
                outputs = self.model(self.reconstructed_data)
                loss_ce = self.loss_fn(outputs, self.target_label.view(-1))
    
                # Compute gradients of loss_ce w.r.t. model parameters
                grads_current = torch.autograd.grad(loss_ce, self.model.parameters(), create_graph=True)
    
                # Compute gradient matching loss with normalized gradients
                grad_diff = 0.0
                for gx, gy in zip(grads_current, normalized_target_grads):
                    gx_norm = gx.norm()
                    gy_norm = gy.norm()
    
                    gx_normalized = gx / (gx_norm + offset)
                    gy_normalized = gy / (gy_norm + offset)
    
                    grad_diff += nn.functional.mse_loss(gx_normalized, gy_normalized)
    
                # add pixel-wise loss
                #pixel_loss = nn.functional.mse_loss(self.reconstructed_data, ground_truth_data)
                total_loss = grad_diff # Adjust weighting as necessary
    
                # Backward on total loss to update reconstructed data
                total_loss.backward()
    
                # Append the loss to the list for external access
                total_loss_list.append(total_loss.item())
    
                return total_loss
    
            # Perform optimization step
            optimizer.step(closure)
    
            # Retrieve the latest total_loss from the list
            if total_loss_list:
                latest_loss = total_loss_list[-1]
                loss_history.append(latest_loss)
            else:
                logging.warning("No loss recorded during this step.")
    
            # Compute PSNR
            mse = torch.mean((self.reconstructed_data - ground_truth_data) ** 2).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            psnr_history.append(psnr)
    
            # Logging every 100 steps and the first step
            if step % 100 == 0 or step == 1:
                logging.info(f"Step [{step}/{num_steps}], PSNR: {psnr:.2f} dB, Loss: {latest_loss:.6f}")
    
            # Early stopping if total loss is sufficiently low
            if len(loss_history) > 0 and loss_history[-1] < 1e-6:
                logging.info(f"[*] Converged at step {step}.")
                break
    
        # After optimization, retrieve the reconstructed data
        recovered_data = self.reconstructed_data.detach().cpu()
    
        # Compute recovered label based on model's prediction
        with torch.no_grad():
            recovered_outputs = self.model(self.reconstructed_data)
            recovered_label = torch.argmax(recovered_outputs, dim=1).item()
    
        logging.info(f"[*] Reconstruction completed. Recovered Label: {recovered_label}")
    
        return recovered_data, recovered_label, loss_history, psnr_history


def apply_weight_update(model, weight_update):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weight_update:
                if param.shape != weight_update[name].shape:
                    logging.warning(f"Skipping parameter '{name}' due to shape mismatch: model has {param.shape}, weight update has {weight_update[name].shape}")
                    continue  # Skip mismatched parameters
                param += weight_update[name].to(param.device)
            else:
                logging.warning(f"Parameter '{name}' not found in weight_update.")


def dlm_attack(pickle_path, ground_truth_dir, output_dir, num_steps=1000, lr=0.1, device=DEVICE):
    if not os.path.isfile(pickle_path):
        logging.error(f"Pickle file not found: {pickle_path}")
        return

    # Create output_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model with the specific supernet architecture
    model = get_supernet()
    model.eval()  # Set to eval mode
    logging.info("[*] Initialized supernet model.")

    # Log requires_grad status
    for name, param in model.named_parameters():
        logging.info(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    # Load weight updates
    weight_updates = load_weight_updates(pickle_path)

    # Load ground truth images (for all labels)
    ground_truth_images_all = load_ground_truth_images(ground_truth_dir)  # Load one image per label

    # Initialize LPIPS loss function once
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # Iterate over each round and perform the attack
    for idx, (round_num, update_path) in enumerate(weight_updates):
        logging.info(f"[*] Processing Round {round_num} ({idx+1}/{len(weight_updates)})")

        if not os.path.exists(update_path):
            logging.error(f"Weight update file not found at '{update_path}'. Skipping Round {round_num}.")
            continue

        # Load the weight update
        try:
            with open(update_path, 'rb') as f:
                weight_update = pickle.load(f)
            logging.info(f"Loaded weight update for Round {round_num} from '{update_path}'.")
        except Exception as e:
            logging.error(f"Error loading weight update for Round {round_num}: {e}")
            continue

        # Apply the weight update to the model
        try:
            apply_weight_update(model, weight_update)
            logging.info(f"Applied weight update for Round {round_num}.")
        except Exception as e:
            logging.error(f"Error applying weight update for Round {round_num}: {e}")
            continue  # Skip this update and proceed to the next

        # For initial focus, process only class 0
        target_label = 0
        ground_truth_data_dict = load_ground_truth_images(ground_truth_dir, target_label=target_label)
        
        if target_label not in ground_truth_data_dict:
            logging.error(f"No ground truth image found for label {target_label} in '{ground_truth_dir}'. Skipping.")
            continue

        ground_truth_data = ground_truth_data_dict[target_label]
        ground_truth_label = torch.tensor([target_label], dtype=torch.long).to(device)

        # Initialize the DLM attack with ground_truth_data
        dlm_attacker = DLM_Attack(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=optim.LBFGS,  # Switch to LBFGS
            optimizer_params={'lr': lr, 'max_iter': 1, 'history_size': 10, 'line_search_fn': 'strong_wolfe'},
            target_label=ground_truth_label,
            ground_truth_data=ground_truth_data,
            device=device
        )

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

        # Compute dummy gradients from a dummy input
        dummy_data = torch.randn_like(ground_truth_data).to(device).requires_grad_(True)
        dummy_label = torch.tensor([target_label], dtype=torch.long).to(device)

        # Forward pass with dummy data
        dummy_outputs = model(dummy_data)
        dummy_loss = nn.CrossEntropyLoss()(dummy_outputs, dummy_label.view(-1))

        # Compute gradients with respect to model parameters
        try:
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        except RuntimeError as e:
            logging.error(f"Error computing dummy gradients: {e}")
            continue

        # Normalize gradients
        dummy_norm = torch.stack([g.norm() for g in dummy_dy_dx]).mean()
        target_norm = torch.stack([g.norm() for g in target_gradients.values()]).mean()

        offset = 1e-8
        if dummy_norm == 0 or target_norm == 0:
            logging.error("Gradient norm is zero. Cannot normalize gradients.")
            continue

        normalized_dummy_grads = [g / (dummy_norm + offset) for g in dummy_dy_dx]
        normalized_target_grads = [g / (target_norm + offset) for g in target_gradients.values()]

        # Perform the DLM attack to reconstruct the input
        try:
            recovered_data, recovered_label, loss_history, psnr_history = dlm_attacker.reconstruct_input(
                normalized_target_grads=normalized_target_grads,
                ground_truth_data=ground_truth_data,
                num_steps=num_steps,
                lr=lr
            )
        except Exception as e:
            logging.error(f"Error during reconstruction in Round {round_num}: {e}")
            continue

        # Compute Metrics
        try:
            accuracy = compute_accuracy(recovered_label, ground_truth_label.item())
            psnr = compute_psnr_metric(recovered_data, ground_truth_data)
            ssim_score = compute_ssim_metric(recovered_data, ground_truth_data)
            lpips_score = compute_lpips_score(recovered_data, ground_truth_data, lpips_fn)
        except Exception as e:
            logging.error(f"Error computing metrics for Round {round_num}: {e}")
            accuracy = 0
            psnr = float('nan')
            ssim_score = float('nan')
            lpips_score = float('nan')

        logging.info(f"Metrics for Round {round_num}, Label {target_label}:")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"PSNR: {psnr:.2f} dB")
        logging.info(f"SSIM: {ssim_score:.4f}")
        logging.info(f"LPIPS: {lpips_score:.4f}")

        # Define unique output path for this round and label
        output_path = os.path.join(output_dir, f'recovered_round_{round_num}_label_{target_label}.pt')

        # Save Recovered Data and Metrics
        try:
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
        except Exception as e:
            logging.error(f"Error saving recovered data for Round {round_num}: {e}")

        closest_label_same, closest_image_same, min_psnr_same = find_closest_image_same_class(
            recovered_data,
            ground_truth_images_all,  # Only class 0
            compute_psnr_metric,
            target_label=target_label
        )

        # Visualize Recovered Data and Closest Images
        visualization_path = os.path.join(output_dir, f'visual_round_{round_num}_label_{target_label}.png')
        try:
            visualize_recovered_data(
                recovered_data,
                ground_truth_data,
                closest_image_same,
                save_path=visualization_path
            )
        except Exception as e:
            logging.error(f"Error during visualization for Round {round_num}: {e}")

        # Plot and save metrics
        try:
            plot_metrics(loss_history, psnr_history, output_dir, round_num, target_label)
        except Exception as e:
            logging.error(f"Error plotting metrics for Round {round_num}: {e}")

    logging.info("[*] DLM attack completed successfully.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dlm_attack.log")
        ]
    )
    

    parser = argparse.ArgumentParser(description='DLM Attack on SuperFedNAS with CIFAR-10')
    
    parser.add_argument('--pickle', type=str, default='./weight_updates/model_paths.pkl',
                        help='Path to weight updates pickle file')
    parser.add_argument('--ground_truth_dir', type=str, default='./ground_truths/',
                        help='Directory containing ground truth image files')
    parser.add_argument('--output_dir', type=str, default='./recovered_data/',
                        help='Directory to save recovered data and metrics')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for optimizer')
    parser.add_argument('--gpu_id', type=int, default=7,
                        help='GPU ID to use (0-7)')
    
    args = parser.parse_args()
    
    global DEVICE
    DEVICE = set_device(args.gpu_id)
    logging.info(f"Using device: {DEVICE} (GPU ID: {args.gpu_id})")
    
    # Perform the DLM attack
    dlm_attack(
        pickle_path=args.pickle,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,           
        num_steps=args.steps,
        lr=args.lr,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
