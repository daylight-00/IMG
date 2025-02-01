import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
from dataprovider import DataProvider
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            X = batch[:-1]
            X = [x.to(device) for x in X]
            y_pred = model(*X).cpu().numpy()
            all_preds.append(y_pred)
    return np.concatenate(all_preds)

def calculate_roc_auc(data_provider, model, config, device):
    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"].get("batch_size", len(dataset))
    num_workers = config["Data"]["num_workers"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y = np.array([sample[-1] for sample in data_provider.samples])
    y_pred = test_model(model, dataloader, device)
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_pr_auc(data_provider, model, config, device):
    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"].get("batch_size", len(dataset))
    num_workers = config["Data"]["num_workers"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y = np.array([sample[-1] for sample in data_provider.samples])
    y_pred = test_model(model, dataloader, device)
    precision, recall, _ = precision_recall_curve(y, y_pred)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

def plot_general_curves(data_provider, model, config, device, plot_path, model_name):
    fpr, tpr, roc_auc = calculate_roc_auc(data_provider, model, config, device)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=1.5, label=f'ROC curve (area = {roc_auc:.2f})', color='lightseagreen')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (All HLAs)')
    plt.legend(loc="lower right")
    save_path = f'{plot_path}/roc_curve-{model_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved as {save_path}\\n")

    precision, recall, pr_auc = calculate_pr_auc(data_provider, model, config, device)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=1.5, label=f'PR curve (area = {pr_auc:.2f})', color='purple')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (All HLAs)')
    plt.legend(loc="lower left")
    save_path = f'{plot_path}/pr_curve-{model_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved as {save_path}\\n")

def plot_top_10_curves(data_provider, model, config, device, plot_path, model_name, DATA_PROVIDER_ARGS):
    top_10_hlas = data_provider.top_10_hlas
    colors = plt.cm.get_cmap('tab10', 10)

    plt.figure(figsize=(8, 6))
    for index, hla in enumerate(top_10_hlas):
        data_provider_hla = DataProvider(**DATA_PROVIDER_ARGS, specific_hla=hla)
        fpr, tpr, roc_auc = calculate_roc_auc(data_provider_hla, model, config, device)
        plt.plot(fpr, tpr, lw=1.5, label=f'{hla} (area = {roc_auc:.2f})', color=colors(index))
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Top 10 HLAs)')
    plt.legend(loc="lower right")
    save_path = f'{plot_path}/roc_curve_hla-{model_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC curve for top 10 HLAs saved as {save_path}")

    plt.figure(figsize=(8, 6))
    for index, hla in enumerate(top_10_hlas):
        data_provider_hla = DataProvider(**DATA_PROVIDER_ARGS, specific_hla=hla)
        precision, recall, pr_auc = calculate_pr_auc(data_provider_hla, model, config, device)
        plt.plot(recall, precision, lw=1.5, label=f'{hla} (area = {pr_auc:.2f})', color=colors(index))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Top 10 HLAs)')
    plt.legend(loc="lower left")
    save_path = f'{plot_path}/pr_curve_hla-{model_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve for top 10 HLAs saved as {save_path}\\n")
