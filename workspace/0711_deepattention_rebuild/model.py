# model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import config
from dataprovider import DataProvider
from tqdm import tqdm

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm_hla = nn.LSTM(385, hidden_dim, batch_first=True)
        self.lstm_pep = nn.LSTM(15, hidden_dim, batch_first=True)
        
        self.attention_hla = nn.Linear(hidden_dim, 1)
        self.attention_pep = nn.Linear(hidden_dim, 1)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, hla_seq, hla_mask, pep_seq, pep_mask):
        hla_out, _ = self.lstm_hla(hla_seq)
        pep_out, _ = self.lstm_pep(pep_seq)
        
        hla_attn_weights = torch.softmax(self.attention_hla(hla_out), dim=1)
        pep_attn_weights = torch.softmax(self.attention_pep(pep_out), dim=1)
        
        hla_attn_applied = torch.bmm(hla_attn_weights.permute(0, 2, 1), hla_out).squeeze(1)
        pep_attn_applied = torch.bmm(pep_attn_weights.permute(0, 2, 1), pep_out).squeeze(1)
        
        combined = torch.cat((hla_attn_applied, pep_attn_applied), dim=1)
        output = self.fc(combined)
        return output

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                (hla_encoded, hla_mask, pep_encoded, pep_mask), labels = batch
                
                # Move tensors to the GPU
                hla_encoded, hla_mask = hla_encoded.to(device), hla_mask.to(device)
                pep_encoded, pep_mask = pep_encoded.to(device), pep_mask.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(hla_encoded, hla_mask, pep_encoded, pep_mask)
                loss = criterion(outputs.squeeze(), labels.float())
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            (hla_encoded, hla_mask, pep_encoded, pep_mask), labels = batch
            
            # Move tensors to the GPU
            hla_encoded, hla_mask = hla_encoded.to(device), hla_mask.to(device)
            pep_encoded, pep_mask = pep_encoded.to(device), pep_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(hla_encoded, hla_mask, pep_encoded, pep_mask)
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            print(f"Predicted: {predicted}, Actual: {labels}")
            break

def main():
    # Configuration
    input_dim = 23  # BLOSUM62 vector size
    hidden_dim = 64
    output_dim = 1
    num_epochs = config["Training"]["epochs"]
    batch_size = config["Training"]["batch_size"]
    learning_rate = 0.001

    # NVIDIA GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 여기서 "0"은 NVIDIA GPU의 장치 인덱스입니다.

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GPU 활성화 여부 출력
    if torch.cuda.is_available():
        print("GPU가 활성화되었습니다.")
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")

    # DataLoader
    data_provider = DataProvider(encoding_method=config["Model"]["encoding_method2"])
    dataloader = DataLoader(data_provider, batch_size=batch_size, shuffle=True)
    
    # Model, loss function and optimizer
    model = AttentionModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)
    
    # Evaluate the model
    evaluate_model(model, dataloader, device)

if __name__ == "__main__":
    main()
