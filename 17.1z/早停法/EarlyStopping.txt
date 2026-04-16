class EarlyStopping:
    def __init__(self, model: torch.nn.Module, patience: int, save_path: str) -> None:
        self.model = model
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None or self.best_loss>=val_loss:
            self.best_loss = val_loss
            self.counter=0
            self.save_model()
        else:
            self.counter+=1
            if self.counter>=self.patience:
                self.early_stop = True 
        #TODO

    def save_model(self) -> None:
        torch.save(self.model.state_dict(),self.save_path)
        #TODO
