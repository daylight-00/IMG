import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
from plot import plot_general_curves, plot_top_10_curves
import h5py
import re

def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def normalize_hla_name(hla_name):
    hla_name = re.sub(r'\*|:|-', '', hla_name)
    return hla_name

def test_model(model, dataloader, device, target_layer=None):
    model.eval()
    # print(model)
    all_preds = []
    all_features = []
    if True:
        features = {}
        def hook_fn(module, input, output):
            # Detach and move to CPU, then convert to numpy
            features['feature'] = output.detach().cpu().numpy()
        # Register the hook
        target_layer = model.self_attn
        hook = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            X = batch[:-1]
            X = [x.to(device) for x in X]
            y_pred = model(*X)
            all_preds.append(y_pred.cpu().numpy())

            if features.get('feature') is not None:
                all_features.append(np.squeeze(features.get('feature')))
    if True:
        hook.remove()

    print(f'Length of all_features: {len(all_features)}')
    print(f'Length of all_preds: {len(all_preds)}')

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_features

def main(config_path):
    config = load_config(config_path)
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)

    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data']['test_path'],
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }

    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config["chkp_path"], f"{model_name}-{config['Test']['chkp_prefix']}.pt")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f'Model loaded on {device}')

    data_provider = DataProvider(**DATA_PROVIDER_ARGS)
    print(f"Samples in test set: {len(data_provider)}")
    if config['Test']['plot']:
        print("Plotting is enabled.")
        plot_general_curves(data_provider, model, config, device, plot_path, model_name)
        plot_top_10_curves(data_provider, model, config, device, plot_path, model_name, DATA_PROVIDER_ARGS)
    else:
        print("Plotting is disabled.")

    if config['Test']['feat_extract'] or config['Test']['save_pred']:
        dataset = config["encoder"](data_provider, **config["encoder_args"])
        batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
        num_workers = config["Data"]["num_workers"]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        y = np.array([sample[-1] for sample in data_provider.samples])  # Optimized data access
        
        target_layer_name = config["Test"]["target_layer"]
        target_layer = dict([*model.named_modules()])[target_layer_name]

        y_pred, features  = test_model(model, dataloader, device, target_layer)

    if config['Test']['feat_extract']:
        print("Feature extraction is enabled.")
        epi_list = [sample[1] for sample in data_provider.samples]
        hla_list = [sample[0] for sample in data_provider.samples]
        epi_hla_list = [f'{epi_list[i]}_{hla_list[i]}' for i in range(len(epi_list))]
        epi_hla_list = [normalize_hla_name(hla) for hla in epi_hla_list]

        save_path = config["Test"]["feat_path"]
        features_path = os.path.join(save_path, f"{model_name}-feat.h5")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with h5py.File(features_path, 'a') as f:
            for i, epi in enumerate(epi_hla_list):
                try:
                    if epi in f:
                        print(f'Dataset {epi} already exists. Skipping.')
                        continue
                    f.create_dataset(epi, data=features[i])
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'Failed to save dataset for {epi}')
                    continue
        print(f"Features saved to {features_path}")

    if config['Test']['save_pred']:
        print("Saving predictions")
        y_pred_save = 'y_pred_save.csv'
        np.savetxt(y_pred_save, y_pred, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()

    main(args.config_path)
