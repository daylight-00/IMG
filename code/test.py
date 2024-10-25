# test.py

import sys, os
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd  # 추가
from sklearn.preprocessing import StandardScaler  
from train import SequenceDataset

def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        with tqdm(dataloader, desc="Testing") as pbar:
            for batch in pbar:
                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                y_pred = model(*X).cpu().numpy()
                all_preds.append(y_pred)
    return np.concatenate(all_preds)

def calculate_roc_auc(data_provider, model, config, device):
    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
    num_workers = config["Data"]["num_workers"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y = np.array([sample[-1] for sample in data_provider.samples])  # Optimized data access
    y_pred = test_model(model, dataloader, device)
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)
    
    # Model loading
    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config["chkp_path"], config["chkp_name"] + '-' + config["Test"]["chkp_prefix"] + '.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f'Model loaded on {device}')

    ########################################################################################
    # General ROC curve

    if 'epi_path' in config["Data"] and 'hla_path' in config["Data"]:
        DATA_PROVIDER_ARGS = {
            "epi_path": config['Data']['test_path'],
            "epi_args": config['Data']['test_args'],
            "hla_path": config['Data']['hla_path'],
            "hla_args": config['Data']['hla_args'],
        }

        # Data loading with DataProvider
        data_provider = DataProvider(**DATA_PROVIDER_ARGS)
        print(f"Samples in test set: {len(data_provider)}")
        
        # General ROC curve
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
        plt.savefig(save_path)
        print(f"ROC curve saved as {save_path}\n")

        # Top 10 HLA-specific ROC curves
        top_10_hlas = data_provider.top_10_hlas
        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('tab10', 10)
        for index, hla in enumerate(top_10_hlas):
            data_provider_hla = DataProvider(**DATA_PROVIDER_ARGS, specific_hla=hla)
            print(f"Samples in test set for {hla}: {len(data_provider_hla)}")
            fpr, tpr, roc_auc = calculate_roc_auc(data_provider_hla, model, config, device)
            plt.plot(fpr, tpr, lw=1.5, label=f'{hla} (area = {roc_auc:.2f})', color=colors(index))
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Top 10 HLAs)')
        plt.legend(loc="lower right")
        save_path = f'{plot_path}/roc_curve_hla-{model_name}.png'
        plt.savefig(save_path)
        print(f"ROC curve saved as {save_path}")

    elif 'hla_embedding_test_path' in config["Data"] and 'epitope_embedding_test_path' in config["Data"]:
        # Load embeddings and metadata
        hla_embeddings = np.load(config["Data"]["hla_embedding_test_path"], allow_pickle=True)
        epitope_embeddings = np.load(config["Data"]["epitope_embedding_test_path"], allow_pickle=True)
        meta_data = pd.read_csv(config["Data"]["meta_data_test_path"])
        targets = meta_data['target'].values.astype(np.float32)

        # Combine embeddings and standardize
        combined_embeddings = np.hstack((hla_embeddings, epitope_embeddings))
        scaler = StandardScaler()
        combined_embeddings = scaler.fit_transform(combined_embeddings)

        # Dataset and DataLoader
        test_dataset = SequenceDataset(combined_embeddings, targets)
        test_loader = DataLoader(test_dataset, batch_size=config["Test"]["batch_size"], shuffle=False)

        # Testing and ROC curve
        y_pred = test_model(model, test_loader, device)
        fpr, tpr, _ = roc_curve(targets, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=1.5, label=f'ROC curve (area = {roc_auc:.2f})', color='royalblue')
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Custom Embeddings)')
        plt.legend(loc="lower right")
        save_path = f'{plot_path}/roc_curve_custom_embeddings-{model_name}.png'
        plt.savefig(save_path)
        print(f"ROC curve for custom embeddings saved as {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()

    main(args.config_path)
