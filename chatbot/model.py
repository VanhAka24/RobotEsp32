import torch
import torch.nn as nn

# Định nghĩa một lớp mạng nơ-ron tùy chỉnh bằng cách kế thừa từ nn.Module
class NeuraNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuraNet, self).__init__()
        # Lớp fully connected đầu tiên: từ đầu vào (input_size) sang tầng ẩn (hidden_size)
        self.l1 = nn.Linear(input_size, hidden_size)
        
        # Lớp fully connected thứ hai: từ tầng ẩn (hidden_size) sang tầng ẩn tiếp theo (hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        
        # Lớp fully connected cuối cùng: từ tầng ẩn (hidden_size) sang đầu ra (num_classes)
        self.l3 = nn.Linear(hidden_size, num_classes)
        
        # Hàm kích hoạt ReLU để áp dụng tính phi tuyến cho đầu ra của mỗi tầng
        self.relu = nn.ReLU()

    def forward(self, x):
        # Truyền dữ liệu qua lớp l1 và áp dụng ReLU
        out = self.l1(x)
        out = self.relu(out)
        
        # Truyền dữ liệu qua lớp l2 và áp dụng ReLU
        out = self.l2(out)
        out = self.relu(out)
        
        # Truyền dữ liệu qua lớp l3 để tạo đầu ra cuối cùng
        out = self.l3(out)
        return out
