import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from math import sqrt

def read_items(file_path):
    i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
              'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = []
    with open(file_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            movie = {col: row[i] if i < len(row) else None for i, col in enumerate(i_cols)}
            items.append(movie)
    return items

# Đọc file u.user
def read_users(file_path):
    users = []
    with open(file_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            users.append({
                'user_id': int(row[0]),
                'age': int(row[1]),
                'sex': row[2],
                'occupation': row[3],
                'zip_code': row[4]
            })
    return users

# Đọc ratings
def read_ratings(file_path):
    ratings = []
    with open(file_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            ratings.append([int(row[0]), int(row[1]), float(row[2])])
    return ratings

# Đọc dữ liệu
items = read_items('C:/ml-100k/u.item')
users = read_users('C:/ml-100k/u.user')
ratings_base = read_ratings('C:/ml-100k/ua.base')
ratings_test = read_ratings('C:/ml-100k/ua.test')

n_users = len(users)
n_items = len(items)
print(len(users), len(items))

X0 = np.array([list(map(int, list(item.values())[5:])) for item in items])
tfidf = TfidfTransformer(smooth_idf=True).fit_transform(X0.tolist()).toarray()

#  Dataset cho CNN
def get_items_rated_by_user(rate_matrix, user_id):
    rate_matrix = np.array(rate_matrix)
    y = rate_matrix[:, 0]
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

class MovieRatingDataset(Dataset):
    def __init__(self, tfidf, ratings, n_users):
        self.samples = []
        for user_id in range(n_users):
            item_ids, scores = get_items_rated_by_user(ratings, user_id)
            for item_id, rating in zip(item_ids, scores):
                self.samples.append((tfidf[int(item_id)], user_id, rating))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, user_id, y = self.samples[idx]
        x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Convert to tensor with a batch dimension
        user_id_tensor = torch.tensor(user_id, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, user_id_tensor, y_tensor

#  Mô hình CNN
class CNNRegressor(nn.Module):
    def __init__(self, input_dim, n_users):
        super(CNNRegressor, self).__init__()

        # Các tầng Conv1D
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.fc = nn.Linear(64, n_users)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool(x).squeeze(2)
        x = self.fc(x)
        return x

# Chuẩn bị dữ liệu
train_dataset = MovieRatingDataset(tfidf, ratings_base, n_users)
test_dataset = MovieRatingDataset(tfidf, ratings_test, n_users)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Huấn luyện
model = CNNRegressor(input_dim=tfidf.shape[1], n_users=n_users)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader):
    model.train()
    total_loss = 0
    for x, user_ids, ratings in loader:
        preds = model(x)
        preds = preds.gather(1, user_ids.unsqueeze(1)).squeeze(1)  # Gather user-specific predictions
        loss = criterion(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, user_ids, ratings in loader:
            preds = model(x)
            preds = preds.gather(1, user_ids.unsqueeze(1)).squeeze(1)
            loss = criterion(preds, ratings)
            total_loss += loss.item() * x.size(0)
    rmse = sqrt(total_loss / len(loader.dataset))
    return rmse

n_epochs = 100  
patience = 5   
best_rmse = float('inf')
patience_counter = 0

for epoch in range(n_epochs):
    train_loss = train(model, train_loader)
    test_rmse = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test RMSE: {test_rmse:.4f}")

    if test_rmse < best_rmse:
        best_rmse = test_rmse
        patience_counter = 0
        
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1

    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load("best_model.pth"))
