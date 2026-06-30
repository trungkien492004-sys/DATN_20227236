import pandas as pd

# Hàm để đọc file nhanh
def load_data(file_name):
    return pd.read_csv(file_name, sep=';', header=None, names=['text', 'emotion'])

# Nạp 3 bộ dữ liệu
train_data = load_data('train.txt')
test_data = load_data('test.txt')
val_data = load_data('val.txt')

# Kiểm tra số lượng dòng của mỗi bộ
print(f"Số dòng tập Train: {len(train_data)}")
print(f"Số dòng tập Test: {len(test_data)}")
print(f"Số dòng tập Validation: {len(val_data)}")