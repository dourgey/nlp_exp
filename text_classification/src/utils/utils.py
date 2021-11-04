class EarlyStopping:
    def __init__(self, max_count=10):
        self.max_count = max_count
        self.min_loss = float("+inf")
        self.count = 0

    def step(self, valid_loss) -> bool:
        if valid_loss < self.min_loss:
            self.count = 0
            self.min_loss = valid_loss
        else:
            self.count += 1
        if self.count > self.max_count:
            return True
        return False