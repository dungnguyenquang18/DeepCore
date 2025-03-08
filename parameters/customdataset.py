import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, x, y, classes=None):
        """
        Args:
            x (numpy.ndarray): Dữ liệu đầu vào có shape (N, H, W, C).
            y (numpy.ndarray): Nhãn dạng one-hot hoặc chỉ số lớp (N,).
            classes (list): Danh sách các nhãn lớp (nếu có).
        """
        self.x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)  # Đổi shape thành (N, C, H, W)
        self.y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)  # Chuyển one-hot thành chỉ số lớp
        self.classes = classes if classes is not None else list(range(len(np.unique(self.y))))  # Tự động tạo lớp nếu không có

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
