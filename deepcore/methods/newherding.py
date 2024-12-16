from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist, cossim
from ..nets.nets_utils import MyDataParallel


def euclid_dist(x, y):
    # Calculate the squared differences
    z = torch.sum((x - y) ** 2, dim=-1)
    # Take the square root to get the Euclidean distance
    z = torch.sqrt(z)
    return z

class NewHerding(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=30,
                 specific_model="ResNet18", balance: bool = False, metric="euclidean", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model, **kwargs)

        self.model = None

        if metric == "cosine_simliarity":
            self.metric = cossim
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
    def __self_attention(self, matrix, num_head=1):
        q, k, v = torch.tensor(matrix), torch.tensor(matrix), torch.tensor(matrix)  # Added value tensor
        out = torch.matmul(q, k.permute(-1, 0)) 
        out = torch.layer_norm(out, normalized_shape=out.shape[-1:])  # Add normalized_shape argument
        return torch.mean(out, dim=0)
    
    

    def herding(self, matrix, budget: int, index=None):
        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = torch.arange(sample_num).to(matrix.device)  # Ensure indices are on the same device as matrix
        with torch.no_grad():
            mu = self.__self_attention(matrix)
            select_result = torch.zeros(sample_num, dtype=torch.bool, device=matrix.device)  # Use the same device

            for i in range(budget):
                if i % self.args.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                dist = []  # Initialize as a list to avoid zero-dimensional tensor issue
                for img in matrix[~select_result]:  # No need to move to CPU
                    possible_select_result = torch.cat((matrix[select_result], img.unsqueeze(0)))  # Use torch.cat instead of np.append
                    dist.append(euclid_dist(mu, self.__self_attention(possible_select_result)))  # Store distances in a list
                
                dist = torch.tensor(dist)  # Convert list to tensor after the loop
                min_index = torch.argmin(dist).item()
                p = torch.where(~select_result)[0][min_index]
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

