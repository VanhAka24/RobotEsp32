import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# Hàm tách câu thành các token (từ hoặc dấu câu riêng lẻ)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Hàm lấy "gốc" của từ (stem) để chuẩn hóa và giảm dạng từ về dạng cơ bản nhất
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())  # Đưa từ về chữ thường trước khi stem

# Hàm tạo bag of words - một vector biểu diễn câu dựa trên từ vựng cho trước
def bag_of_words(tokenized_sentence, all_words):
    # Đưa các từ trong câu đã tách về dạng gốc
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    # Khởi tạo một vector toàn số 0 với kích thước bằng số lượng từ trong `all_words`
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # Gán giá trị 1 cho các từ xuất hiện trong câu
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
