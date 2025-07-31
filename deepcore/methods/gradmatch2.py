import torch
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls
from .earlytrain import EarlyTrain
from ..nets.nets_utils import MyDataParallel


# https://github.com/krishnatejakk/GradMatch

class GradMatch2(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance=False, dst_val=None, lam: float = 1., **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)
        self.balance = balance
        self.dst_val = dst_val

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))
            
            




    def herding(self, matrix, mu, budget: int, index=None):
        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad(): 
            select_result = np.zeros(sample_num, dtype=bool)
            
            for i in range(budget):
                if i % self.args.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                    
                dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                    matrix[~select_result])

                p = torch.argmin(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True
                        
                
                
                
        if index is None:
            index = indices
        return index[select_result]

    def normalize_and_topk_indices(self, A: torch.Tensor, b: torch.Tensor, budget: int, device='cuda') -> torch.Tensor:
        """
        Dịch A để mean = b, sau đó chọn chỉ số của budget điểm gần b nhất trong A'.

        Args:
            A (torch.Tensor): ma trận (d, n)
            b (torch.Tensor): vector đích (d,)
            budget (int): số lượng điểm cần chọn
            device (str): 'cuda' hoặc 'cpu'

        Returns:
            torch.Tensor: tensor 1D chứa chỉ số (index) trong A gốc, kích thước (budget,)
        """
        A = A.to(device)
        b = b.to(device)

        # 1. Dịch mean của A về b
        mean_A = A.mean(dim=1, keepdim=True)        # (d, 1)
        shift = b.view(-1, 1) - mean_A               # (d, 1)
        A_shifted = A + shift                        # (d, n)

        # 2. Tính khoảng cách L2 từ mỗi điểm tới b
        dists = torch.norm(A_shifted - b.view(-1, 1), dim=0)  # (n,)

        # 3. Trả về chỉ số top-k
        topk_indices = torch.topk(dists, k=budget, largest=False).indices  # (budget,)
        return topk_indices






    def calc_gradient(self, index=None, val=False):
        self.model.eval()
        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=self.args.selection_batch, num_workers=self.args.workers)
            sample_num = len(self.dst_val.targets) if index is None else len(index)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch, num_workers=self.args.workers)
            sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features
        gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)],
                                requires_grad=False, device=self.args.device)

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device)).requires_grad_(True)
            loss = self.criterion(outputs, targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs, retain_graph=True)[0].cpu()
                weight_parameters_grads = self.model.embedding_recorder.embedding.cpu().view(batch_num, 1,
                                                    self.embedding_dim).repeat(1,self.args.num_classes,1) *\
                                                    bias_parameters_grads.view(batch_num, self.args.num_classes,
                                                    1).repeat(1, 1, self.embedding_dim)
                gradients[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] =\
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)

        return gradients

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        self.model.no_grad = True
        with self.model.embedding_recorder:
            if self.dst_val is not None:
                val_num = len(self.dst_val.targets)

            if self.balance:
                selection_result = np.array([], dtype=np.int64)
                weights = np.array([], dtype=np.float32)
                for c in range(self.args.num_classes):
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    cur_gradients = self.calc_gradient(class_index)
                    if self.dst_val is not None:
                        # Also calculate gradients of the validation set.
                        val_class_index = np.arange(val_num)[self.dst_val.targets == c]
                        cur_val_gradients = torch.mean(self.calc_gradient(val_class_index, val=True), dim=0)
                    else:
                        cur_val_gradients = torch.mean(cur_gradients, dim=0)
                    
                    selection_result = self.normalize_and_topk_indices(cur_gradients.T,
                                                                        cur_val_gradients,
                                                                    budget=round(len(class_index) * self.fraction), device=self.args.device)
                    

            else:
                cur_gradients = self.calc_gradient()
                if self.dst_val is not None:
                    # Also calculate gradients of the validation set.
                    cur_val_gradients = torch.mean(self.calc_gradient(val=True), dim=0)
                else:
                    cur_val_gradients = torch.mean(cur_gradients, dim=0)
                selection_result = self.normalize_and_topk_indices(cur_gradients.T,
                                                                cur_val_gradients,
                                                                budget=round(self.n_train * self.fraction), device=self.args.device)

        self.model.no_grad = False
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

