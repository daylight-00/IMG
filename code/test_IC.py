import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            X, Y = batch[:-1], batch[-1]
            X = [x.to(device) for x in X]
            Y = Y.to(device)

            # Forward pass
            y_pred = model(*X).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(Y.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets


def calculate_metrics(predictions, targets):
    """Calculate regression metrics: MSE, MAE, R²."""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    ssr = np.sum((predictions - targets) ** 2)
    sst = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ssr / sst)
    return mse, mae, r2


def plot_predictions_vs_targets(predictions, targets, save_path):
    """Plot predictions vs targets for regression evaluation."""
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.6, color='blue', label="Predictions")
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color='red', linestyle='--', label="Ideal")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predictions vs True Values")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main(config_path):
    # Load configuration
    config = load_config(config_path)
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)

    # Data loading
    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data']['test_path'],
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }

    data_provider = DataProvider(**DATA_PROVIDER_ARGS)
    print(f"Samples in test set: {len(data_provider)}")

    # Create test dataset and DataLoader
    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
    num_workers = config["Data"]["num_workers"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load model
    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config["chkp_path"], config["chkp_name"] + '-' + config["Test"]["chkp_prefix"] + '.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f'Model loaded on {device}')

    # Testing and metrics calculation
    predictions, targets = test_model(model, dataloader, device)
    mse, mae, r2 = calculate_metrics(predictions, targets)

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    # Plot predictions vs targets
    plot_save_path = f"{plot_path}/predictions_vs_targets-{model_name}.png"
    plot_predictions_vs_targets(predictions, targets, plot_save_path)
    print(f"Prediction vs True Values plot saved as {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()

    main(args.config_path)
