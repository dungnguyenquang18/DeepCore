class Args:
    
    def __init__(self, model='ResNet18', channel=3, num_classes=43, im_size=(32, 32), selection_batch=128,
                 print_freq=100, workers=4, device='cuda', selection_method="LeastConfidence",
                 selection_optimizer="Adam", selection_lr=1e-3, selection_momentum=0.0, selection_weight_decay=1e-5,selection_nesterov=False, **kwargs):
        self.model = model
        self.channel = channel
        self.num_classes = num_classes
        self.im_size = im_size
        self.selection_batch = selection_batch
        self.print_freq = print_freq
        self.device = device
        self.selection_method = selection_method
        self.selection_optimizer = selection_optimizer
        self.selection_lr = selection_lr
        self.selection_nesterov = selection_nesterov
        self.selection_momentum = selection_momentum
        self.selection_weight_decay = selection_weight_decay
        if device == 'cuda': 
            self.gpu = kwargs.get('gpu', None)
            self.workers = workers
        else:
            self.gpu = None
            self.workers = 0