import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataprovider import DataProvider
from encoder import encode_data
import torch.nn.functional as F
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LeNetConvPoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LeNetConvPoolLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return F.relu(self.conv(x))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer0 = LeNetConvPoolLayer(1, 50, (5, 183))
        self.layer1 = LeNetConvPoolLayer(50, 10, (5, 183))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(1 * 5 * 10, 50)
        self.fc2 = nn.Linear(50, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def train_model(model, train_loader, val_loader, criterion, optimizer, device, log_file, num_epochs=100, patience=10):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - [Training process]: [base_model0]-[model0]-[Epoch %(epoch)04d] - time: %(time)3d s, train_acc: %(train_acc).5f, val_acc: %(val_acc).5f, train_loss: %(train_loss).5f, val_loss: %(val_loss).5f')

    for epoch in range(num_epochs):
        if early_stop:
            print(f'Early stopping at epoch {epoch}')
            break

        start_time = time.time()
        epoch_loss = 0
        val_loss = 0
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0

        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

                batch_matrices, affinities = batch
                batch_matrices = batch_matrices.to(device)
                affinities = affinities.to(device)

                optimizer.zero_grad()
                outputs = model(batch_matrices)
                loss = criterion(outputs, affinities)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

                # Accuracy 계산
                predicted = (outputs > 0.5).float()
                correct_train += (predicted == affinities).sum().item()
                total_train += affinities.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch_matrices, affinities = batch
                batch_matrices = batch_matrices.to(device)
                affinities = affinities.to(device)
                
                outputs = model(batch_matrices)
                loss = criterion(outputs, affinities)
                val_loss += loss.item()

                # Accuracy 계산
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == affinities).sum().item()
                total_val += affinities.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        epoch_time = int(time.time() - start_time)
        logging.info('', extra={'epoch': epoch, 'time': epoch_time, 'train_acc': train_acc, 'val_acc': val_acc, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

        # Early stopping and best model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "bestmodel.pytorch")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

if __name__ == "__main__":
    # DataProvider 객체 생성
    data_provider = DataProvider()
    x, y = encode_data(data_provider)

    # 데이터를 학습용과 검증용으로 나누기 (클래스 비율 유지)
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

    # numpy 배열을 텐서로 변환
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # TensorDataset으로 변환
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)

    # DataLoader를 사용하여 데이터를 배치로 로드
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)  # L2 정규화 추가

    # 로그 파일 설정
    log_file = "log.txt"

    # 모델 학습
    train_model(model, train_loader, val_loader, criterion, optimizer, device, log_file, num_epochs=100)
