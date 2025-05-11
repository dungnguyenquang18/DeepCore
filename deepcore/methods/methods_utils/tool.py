import numpy as np
import torch
from .ellipsoid import EllipsoidND
import cvxpy as cp
import numpy as np
from numpy.linalg import eig, inv
from scipy.spatial import ConvexHull

def compute_rank(matrix):
    """
    Compute the rank of a matrix.

    Parameters:
    matrix (numpy array): Input matrix

    Returns:
    int: Rank of the matrix
    """
    # Replace this with your actual compute_rank function
    return np.linalg.matrix_rank(matrix)



def pca_reduce(X: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Giảm chiều dữ liệu sử dụng PCA, chỉ trả về dữ liệu giảm chiều.
    
    Args:
        X (torch.Tensor): Dữ liệu đầu vào, shape (n_samples, n_features).
        n_components (int): Số chiều mong muốn sau khi giảm.
    
    Returns:
        torch.Tensor: Dữ liệu sau khi giảm chiều, shape (n_samples, n_components).
    
    Raises:
        ValueError: Nếu X không phải tensor 2D, n_components không hợp lệ,
                    hoặc n_components lớn hơn số đặc trưng.
    """
    # Kiểm tra đầu vào
    if not isinstance(X, torch.Tensor) or X.ndim != 2:
        raise ValueError("X must be a 2D torch.Tensor")
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    if n_components > X.shape[1]:
        raise ValueError("n_components must not exceed number of features")
    if X.shape[0] == 0:
        return torch.empty(0, n_components)

    # Chuẩn hóa dữ liệu: trừ trung bình
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Tính ma trận hiệp phương sai
    covariance_matrix = torch.cov(X_centered.T)

    # Phân tích giá trị riêng
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Sắp xếp theo giá trị riêng giảm dần
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]

    # Chọn n_components vector riêng
    top_eigenvectors = eigenvectors[:, :n_components]

    # Chiếu dữ liệu
    X_reduced = torch.matmul(X_centered, top_eigenvectors)

    return X_reduced

def compute_P_prime(P: torch.Tensor):
    """
    Projects a set of points P onto their affine subspace and returns:
    - P_prime: the coordinates of P in the affine subspace (lower dimension)
    - mapping: a list of tuples (original_index, projected_index)

    Args:
        P (np.ndarray): shape (n_points, d), input points

    Returns:
        P_prime (np.ndarray): shape (n_points, k), projected points in affine subspace
        mapping (list): list of tuples (original_index, projected_index)
    """
    P = np.asarray(P)
    mean = np.mean(P, axis=0)
    P_centered = P - mean

    # Use the provided compute_rank function
    rank = compute_rank(P_centered)
    U, S, Vt = np.linalg.svd(P_centered, full_matrices=False)
    Y = Vt[:rank].T  # d x k

    P_prime = P_centered @ Y
    mapping = [(i, i) for i in range(P.shape[0])]

    return P_prime, mapping


def compute_mvee(P, tol=1e-5, max_iter=1000):
    """
    Tính toán Minimum Volume Enclosing Ellipsoid (MVEE) cho tập điểm P (n điểm, d chiều).
    Trả về một đối tượng EllipsoidND.

    Args:
        P (np.ndarray): shape (n_points, d), tập các điểm.
        tol (float): ngưỡng hội tụ.
        max_iter (int): số vòng lặp tối đa.

    Returns:
        EllipsoidND: ellipsoid bao ngoài tối tiểu.
    """
    P = np.asarray(P)
    n_points, d = P.shape

    # Khởi tạo trọng số đều
    u = np.ones(n_points) / n_points

    for _ in range(max_iter):
        # Tính ma trận X(u)
        X = (P.T * u) @ P
        X_inv = np.linalg.inv(X)

        # Tính giá trị M_i cho từng điểm
        M = np.einsum('ij,jk,ik->i', P, X_inv, P)

        # Tìm chỉ số có M lớn nhất
        j = np.argmax(M)
        max_M = M[j]

        # Kiểm tra hội tụ
        if max_M - d <= tol:
            break

        # Cập nhật trọng số
        step_size = (max_M - d - 1) / ((d + 1) * (max_M - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        u = new_u

    # Tính toán ma trận G và tâm c
    X = (P.T * u) @ P
    c = P.T @ u
    G = np.linalg.inv(X)

    return EllipsoidND(G, c)

# ...existing code...

from scipy.spatial import ConvexHull
from scipy.optimize import linprog



def caratheodory_set(v, P):
    n, d = P.shape
    c = np.zeros(n)
    A_eq = np.vstack([P.T, np.ones(n)])
    b_eq = np.append(v, 1)
    bounds = [(0, 1)] * n
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        lambdas = res.x
        indices = np.where(lambdas > 1e-6)[0]
        if len(indices) > d+1:
            top = np.argsort(lambdas[indices])[-(d+1):]
            indices = indices[top]
        return indices, lambdas[indices]
    else:
        return None, None

def l_infinity_coreset(P, device=None):
    """
    Tính toán coreset cho bài toán MVEE trong không gian l-infinity.
    
    Args:
        P (torch.Tensor): Mảng shape (n_points, d), các điểm đầu vào.
        device: Device để tính toán (CPU/GPU)
    
    Returns:
        np.ndarray: Mảng shape (m, d), coreset cho bài toán 
    """
    if device is None:
        device = P.device
        
    # Giảm chiều dữ liệu
    P_reduced = pca_reduce(P, 50)
    
    # Chuyển về CPU cho các phép tính numpy
    P_cpu = P_reduced.cpu().numpy()
    P_prime, mapping = compute_P_prime(P_cpu)
    S = []

    # Tính MVEE trên CPU
    big_ellipsoid = compute_mvee(P_prime)
    small_ellipsoid = big_ellipsoid.shrink_ellipsoid(compute_rank(P_prime))
    vertices = small_ellipsoid.vertices
    
    # Tính ConvexHull trên CPU
    try:
        hull = ConvexHull(P_prime)
        P_hull = P_prime[hull.vertices]
        
        # Xử lý Carathéodory sets
        for i, v in enumerate(vertices):
            idxs, lambdas = caratheodory_set(v, P_hull)
            if idxs is not None:
                # Chuyển indices về tập gốc thông qua hull.vertices
                original_indices = hull.vertices[idxs]
                S.extend(original_indices)
        
        S = list(set(S))
        
    except Exception as e:
        print(f"Warning: ConvexHull computation failed: {e}")
        # Fallback: sử dụng tất cả các điểm nếu ConvexHull fails
        S = list(range(len(P_prime)))
    
    # Chuyển kết quả về tensor trên device chỉ định
    S_tensor = torch.tensor(S, device=device)
    
    return S_tensor


def ellipsoid_cathedory_coreset(P: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tính toán coreset cho bài toán MVEE trong không gian l-infinity.
    
    Args:
        P (torch.Tensor): Mảng shape (n_points, d), các điểm đầu vào.
        m (int): Kích thước của coreset.
    
    Returns:
        torch.Tensor: Mảng shape (m, d), coreset cho bài toán.
        torch.Tensor: Điểm sensitive của các điểm.
    """
    # Kiểm tra đầu vào


    Q = P.clone()
    s = torch.zeros(P.shape[0], dtype=torch.float32)
    indices = torch.arange(P.shape[0])  # Theo dõi index gốc trong P
    
    i = 1
    l = Q.shape[0]
    r = compute_rank(Q)
    condition = 2 * (r ** 2)
    
    while l >= condition:
        print(f"loop {i}:")
        # Tìm coreset (S là tập index trong Q)
        S = l_infinity_coreset(Q)

        
        # Tính điểm sensitive cho các điểm trong S
        current_rank = compute_rank(Q)
        sensitive_score = current_rank / i
        s[indices[S]] = sensitive_score  # Gán dựa trên index gốc trong P
        
        # Tạo mask để loại bỏ các điểm trong S
        mask = torch.ones(Q.size(0), dtype=torch.bool)
        mask[S] = False
        
        # Cập nhật Q và indices
        Q = Q[mask]
        indices = indices[mask]
        
        l = Q.shape[0]
        r = compute_rank(Q)
        condition = 2 * (r ** 2)
        i += 1
    
    # Tính điểm sensitive cho các điểm còn lại trong Q
    if Q.shape[0] > 0:
        s[indices] = current_rank / i  # Gán dựa trên index gốc trong P
    
    # Chuẩn hóa điểm sensitive
    if s.sum() != 0:
        s = s / s.sum()
    else:
        s = torch.ones_like(s) / s.shape[0]  # Phân phối đều nếu tổng bằng 0
    
    # Chọn m điểm có điểm sensitive cao nhất
    _, top_indices = torch.topk(s, min(m, P.shape[0]))
    
    return top_indices
        
    
