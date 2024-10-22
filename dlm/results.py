import torch
import matplotlib.pyplot as plt
from dlm_attack import visualize_recovered_data

ground_truth = torch.load('ground_truth_round_1.pt')['data']

# Load recovered data
recovered = torch.load('recovered_data_round_1.pt')
recovered_data = recovered['data']
recovered_label = recovered['label']
metrics = recovered['metrics']

print(f"Recovered Label: {recovered_label}")
print(f"Metrics: {metrics}")

# Visualize
visualize_recovered_data(recovered_data, ground_truth, save_path='visual_round_1_client_1.png')



