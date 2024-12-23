import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuraNet

# Tải dữ liệu từ intents.json
# intents.json chứa các mẫu câu và nhãn để mô hình học
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Khởi tạo danh sách từ vựng và nhãn
all_words = []  # Danh sách từ vựng xuất hiện trong tất cả các mẫu câu
tags = []       # Danh sách các nhãn (tags)
xy = []         # Danh sách lưu các cặp (mẫu câu, nhãn)

# Duyệt qua từng intent để tách từ và lưu các cặp (mẫu câu, nhãn)
for intent in intents['intents']:
    tag = intent['tag']  # Lấy nhãn của intent
    tags.append(tag)     # Thêm nhãn vào danh sách tags
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tách từ từ câu
        all_words.extend(w)    # Thêm từ vào danh sách từ vựng
        xy.append((w, tag))    # Thêm cặp (mẫu câu, nhãn) vào xy

# Loại bỏ các từ không cần thiết và chuẩn hóa từ vựng
ignore_words = ["!", "?", ".", ","]  # Các ký tự cần loại bỏ
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Chuẩn hóa và loại bỏ ký tự
all_words = sorted(set(all_words))  # Loại bỏ từ trùng lặp và sắp xếp
tags = sorted(set(tags))  # Sắp xếp danh sách nhãn

# Chuẩn bị dữ liệu huấn luyện
X_train = []  # Dữ liệu đầu vào (bag-of-words)
Y_train = []  # Nhãn tương ứng

# Tạo vector bag-of-words cho mỗi mẫu câu và gán nhãn
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # Tạo vector bag-of-words
    X_train.append(bag)  # Thêm vector vào danh sách
    label = tags.index(tag)  # Gán nhãn dưới dạng số
    Y_train.append(label)  # Thêm nhãn vào danh sách

X_train = np.array(X_train)  # Chuyển đổi sang mảng numpy
Y_train = np.array(Y_train)

# Định nghĩa Dataset
# Lớp Dataset giúp quản lý dữ liệu huấn luyện
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)  # Tổng số mẫu
        self.x_data = X_train  # Dữ liệu đầu vào
        self.y_data = Y_train  # Nhãn

    def __getitem__(self, index):
        # Trả về một mẫu dữ liệu (bag-of-words, nhãn)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # Trả về số lượng mẫu
        return self.n_samples

# Các tham số huấn luyện
batch_size = 8  # Số mẫu xử lý trong một batch
hidden_size = 8  # Kích thước lớp ẩn
output_size = len(tags)  # Số nhãn (đầu ra)
input_size = len(X_train[0])  # Kích thước vector bag-of-words (đầu vào)
learning_rate = 0.001  # Tốc độ học
num_epochs = 1000  # Số lần huấn luyện toàn bộ dữ liệu

# Tạo DataLoader để duyệt qua dữ liệu theo batch
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Kiểm tra thiết bị (GPU hoặc CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình Neural Network
model = NeuraNet(input_size, hidden_size, output_size).to(device)

# Hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()  # Hàm tính lỗi CrossEntropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Bộ tối ưu Adam

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # Chuyển dữ liệu đầu vào sang thiết bị
        labels = labels.to(device, dtype=torch.int64)  # Chuyển nhãn sang thiết bị

        # Tiến trình xuôi (forward pass)
        outputs = model(words)
        loss = criterion(outputs, labels)  # Tính lỗi

        # Tiến trình ngược (backward pass) và cập nhật tham số
        optimizer.zero_grad()  # Xóa gradient cũ
        loss.backward()  # Tính gradient
        optimizer.step()  # Cập nhật tham số

    # In thông tin sau mỗi 100 epoch
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

# Lưu mô hình đã huấn luyện
data = {
    "model_stage": model.state_dict(),  # Trạng thái mô hình
    "input_size": input_size,  # Kích thước đầu vào
    "output_size": output_size,  # Kích thước đầu ra
    "hidden_size": hidden_size,  # Kích thước lớp ẩn
    "all_words": all_words,  # Danh sách từ vựng
    "tags": tags  # Danh sách nhãn
}

FILE = "data.pth"  # Tên file lưu trữ
torch.save(data, FILE)  # Lưu dữ liệu
print(f"Training complete. Model saved to {FILE}")  # Thông báo hoàn thành
