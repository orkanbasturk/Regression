import os
import time
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tkinter import Tk
from tkinter.filedialog import askopenfilename


print("Lütfen veri dosyasını seçin.")
time.sleep(3)
root = Tk()
root.withdraw()  
file_path = askopenfilename(title="Veri dosyasını seçin", filetypes=[("CSV Files", "*.csv")])
if not file_path:
    print("Dosya seçilmedi, çıkılıyor.")
    exit()

try:
    with open(file_path, 'r') as f:
        first_line = f.readline()
        sep = ',' if ',' in first_line else ';'
    data = pd.read_csv(file_path, sep=sep)
except FileNotFoundError:
    print(f"Dosya bulunamadı: {file_path}")
    exit()
except Exception as e:
    print(f"Bir hata oluştu: {e}")
    exit()

print("Veri setindeki sütunlar:")
print(data.columns)
target_column = input("Hedef sütunun adı nedir? ")

data = data.dropna()  
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x = data.drop(columns=[target_column]).values  
y = data[target_column].values  

x = scaler_x.fit_transform(x)  
y = scaler_y.fit_transform(y.reshape(-1, 1)) 

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(x, y)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = random_split(dataset, [train_size, test_size])

batch_size = 32
data_ld = DataLoader(train, batch_size)

model = nn.Linear(x.shape[1], 1) 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
epoch_size = int(input("epoch girin:"))
patience = 500
losses = []
best_loss = float('inf')
patience_counter = 0

for epoch in range(epoch_size):
    for xd, yd in data_ld:
        preds = model(xd)
        loss = F.mse_loss(preds, yd)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    losses.append(loss.item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Erken durdurma! En iyi loss: {best_loss} (Epoch: {epoch + 1})")
        break

    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

test_x, test_y = zip(*test)
test_x = torch.stack(test_x)
test_y = torch.stack(test_y)

test_preds = model(test_x).detach().numpy()
test_y = test_y.numpy()

test_preds = scaler_y.inverse_transform(test_preds)
test_y = scaler_y.inverse_transform(test_y)

mse = mean_squared_error(test_y, test_preds)
r2 = r2_score(test_y, test_preds)

print("\nTest Sonuçları:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Skoru: {r2}")

correlation = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")  
plt.title("Correlation Matrix")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/correlation_matrix.jpg")
plt.close()

plt.figure()
plt.plot(range(len(losses)), losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.savefig("output/loss_graph.jpg")
plt.close()

explainer = shap.Explainer(lambda val: model(torch.tensor(val, dtype=torch.float32)).detach().numpy(), x.numpy())
shap_values = explainer(x.numpy())
plt.figure()
shap.summary_plot(shap_values, x.numpy(), feature_names=data.drop(columns=[target_column]).columns, show=False)
plt.savefig("output/shap_summary.jpg")
plt.close()

while True:
    answer = input("Tahmin yapmak ister misiniz? (Evet/Hayır): ").strip().lower()
    if answer == "evet":
        inputs = []
        for i, feature in enumerate(data.drop(columns=[target_column]).columns):
            value = float(input(f"{feature} için bir değer girin: "))
            inputs.append(value)

        input_tensor = torch.tensor(scaler_x.transform([inputs]), dtype=torch.float32)
        prediction = scaler_y.inverse_transform(model(input_tensor).detach().numpy())
        print(f"Tahmin edilen değer: {prediction[0][0]}")
    elif answer == "hayır":
        print("Uygulama kapatılıyor...")
        print("iletişim: basturkorkan@gmail.com")
        break
    else:
        print("Geçersiz cevap, lütfen 'Evet' veya 'Hayır' yazınız.")
