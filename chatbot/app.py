from flask import Flask, render_template, request, jsonify  # Flask để xây dựng ứng dụng web
import torch  # PyTorch để thực hiện các tác vụ học máy
import json  # Thư viện để làm việc với dữ liệu JSON
from model import NeuraNet  # Lớp mô hình mạng nơ-ron đã được định nghĩa
from nltk_utils import bag_of_words, tokenize  # Các hàm xử lý ngôn ngữ từ NLTK
import random  # Thư viện chọn ngẫu nhiên

app = Flask(__name__)  # Khởi tạo ứng dụng Flask

# Kiểm tra nếu có GPU hỗ trợ, nếu không thì dùng CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đọc file intents.json để lấy thông tin các intent và phản hồi
with open('intents.json', 'r') as f:
    intents = json.load(f)  # Tải file JSON chứa dữ liệu các intents và responses

# Đọc model đã được huấn luyện trước
FILE = 'data.pth'  # Đường dẫn tới file chứa trọng số của mô hình
data = torch.load(FILE, weights_only=True)  # Tải trọng số của mô hình từ file

# Lấy thông tin về các tham số của mô hình từ file
input_size = data["input_size"]  # Kích thước của đầu vào
hidden_size = data["hidden_size"]  # Kích thước của lớp ẩn
output_size = data["output_size"]  # Kích thước của đầu ra
all_words = data["all_words"]  # Danh sách tất cả các từ
tags = data["tags"]  # Danh sách các nhãn (tags) của các intent
model_stage = data["model_stage"]  # Trạng thái mô hình (weights)

# Khởi tạo mô hình mạng nơ-ron với các tham số đã được tải
model = NeuraNet(input_size, hidden_size, output_size).to(device)  # Di chuyển mô hình tới device (GPU hoặc CPU)
model.load_state_dict(model_stage)  # Tải trọng số vào mô hình
model.eval()  # Đặt mô hình vào chế độ đánh giá (không cập nhật trọng số trong quá trình dự đoán)

# Định nghĩa route chính cho trang web
@app.route('/')
def index():
    return render_template('index.html')  # Trả về file HTML cho trang chủ

# Định nghĩa route xử lý chat
@app.route('/chat', methods=['POST'])  # Lắng nghe yêu cầu POST từ người dùng
def chat():
    user_input = request.json['message']  # Lấy đầu vào của người dùng từ JSON
    tokenized_sentence = tokenize(user_input)  # Tokenize câu người dùng nhập
    X = bag_of_words(tokenized_sentence, all_words)  # Chuyển câu đã token thành vector đặc trưng
    X = torch.from_numpy(X).to(device)  # Chuyển đổi numpy array thành tensor và đưa vào device

    output = model(X)  # Dự đoán đầu ra từ mô hình
    
    print(f"Output shape: {output.shape}")  # In ra kích thước của đầu ra từ mô hình

    if output.dim() == 1:  # Nếu đầu ra chỉ có 1 chiều, thêm chiều còn lại để phù hợp với hàm softmax
        output = output.unsqueeze(0)

    probs = torch.softmax(output, dim=1)  # Tính xác suất của các nhãn
    _, predicted = torch.max(probs, dim=1)  # Chọn nhãn có xác suất cao nhất

    tag = tags[predicted.item()]  # Lấy nhãn (tag) từ danh sách tags
    prob = probs[0][predicted.item()]  # Lấy xác suất của nhãn dự đoán

    # Nếu xác suất dự đoán lớn hơn 75%, trả về phản hồi tương ứng với tag
    if prob.item() > 0.75:
        for intent in intents["intents"]:  # Duyệt qua các intents
            if tag == intent["tag"]:  # Nếu tag trùng khớp với tag của intent
                response = random.choice(intent['responses'])  # Chọn ngẫu nhiên một phản hồi từ list
                return jsonify({"response": response})  # Trả về phản hồi cho người dùng
    else:
        return jsonify({"response": "I don't understand..."})  # Nếu không hiểu, trả về thông báo không hiểu

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)  # Chạy ứng dụng Flask với chế độ debug
