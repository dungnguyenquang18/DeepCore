from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel
from sklearn.ensemble import IsolationForest


class Herding(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model="ResNet18", balance: bool = False, metric="euclidean", 
                 use_weights=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model, **kwargs)

        self.use_weights = use_weights
        
        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda: self.finish_run()

            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index), num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)

            self.construct_matrix = _construct_matrix

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                            torch.utils.data.Subset(self.dst_train, index),
                                            batch_size=self.args.selection_batch,
                                            num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.model.embedding_recorder.embedding

        self.model.no_grad = False
        return matrix

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def compute_weights(self, index=None):
        """Tính trọng số w(x) dựa trên Isolation Forest"""
        # Lấy dữ liệu để tính trọng số
        if index is None:
            data_loader = torch.utils.data.DataLoader(
                self.dst_train, batch_size=self.n_train, num_workers=self.args.workers)
        else:
            data_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.dst_train, index),
                batch_size=len(index), num_workers=self.args.workers)
        
        inputs, _ = next(iter(data_loader))
        inputs_np = inputs.flatten(1).cpu().numpy()
        
        # Huấn luyện Isolation Forest
        iso_forest = IsolationForest(random_state=self.random_seed, n_jobs=-1)
        iso_forest.fit(inputs_np)
        
        # Tính điểm bất thường và chuyển thành trọng số
        # AnomalyScore(x) ∈ [0, 1], w(x) = 1 - AnomalyScore(x)
        anomaly_scores = iso_forest.score_samples(inputs_np)
        # Chuẩn hóa về [0, 1]
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        weights = 1 - anomaly_scores
        
        return 1 / torch.from_numpy(weights).float().to(self.args.device)

    def herding(self, matrix, budget: int, index=None):
        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            # Tính trọng số nếu cần
            if self.use_weights:
                weights = self.compute_weights(index)
                # Áp dụng trọng số vào ma trận
                weighted_matrix = matrix * weights.unsqueeze(1)
                mu = torch.sum(weighted_matrix, dim=0) / torch.sum(weights)
            else:
                mu = torch.mean(matrix, dim=0)
                
            select_result = np.zeros(sample_num, dtype=bool)

            for i in range(budget):
                if i % self.args.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                
                if self.use_weights:
                    # Tính tổng có trọng số của các điểm đã chọn
                    if np.any(select_result):
                        selected_sum = torch.sum(matrix[select_result] * 
                                               weights[select_result].unsqueeze(1), dim=0)
                        selected_weight_sum = torch.sum(weights[select_result])
                    else:
                        selected_sum = torch.zeros_like(mu)
                        selected_weight_sum = 0
                    
                    # Tính khoảng cách có trọng số
                    target = ((i + 1) * mu - selected_sum).view(1, -1)
                    
                    # Áp dụng trọng số cho các điểm chưa chọn
                    unselected_weights = weights[~select_result]
                    dist = self.metric(target, matrix[~select_result]) / unselected_weights.unsqueeze(0)
                else:
                    dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                      matrix[~select_result])
                
                p = torch.argmin(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True
                
        if index is None:
            index = indices
        return index[select_result]

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]

                selection_result = np.append(selection_result, self.herding(self.construct_matrix(class_index),
                        budget=round(self.fraction * len(class_index)), index=class_index))
        else:
            selection_result = self.herding(self.construct_matrix(), budget=self.coreset_size)
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
