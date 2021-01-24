import torch

class MultipleChoiceLossCompute:
    "A Loss compute and train function for multiple choice tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, :, 1:, 0].contiguous().view(-1)  # Shape: 252
            M = M.view(-1, M.size(2))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0) * X.size(1), X.size(2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

class ClassificationLossCompute:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion  = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef       = lm_coef
        self.opt           = opt

    def __call__(self, X, Y, M, clf_logits, batch_num=3, accum_steps=10, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        #if lm_logits is not None:
        #    x_shifted = X[:, 1:, 0].contiguous().view(-1)
        #    M         = M.view(-1, M.size(-1))
        #    lm_losses = self.lm_criterion(lm_logits, x_shifted)
        #    lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
        #    lm_losses = lm_losses * M[:, 1:]
        #    lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses
        self.lm_coef =  0
        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        print(train_loss)
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

class SummarizationLossCompute:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion,  opt=None):
        self.lm_criterion  = lm_criterion
        self.opt           = opt

    def __call__(self, X, M, lm_logits, batch_num=None, only_return_losses=False,accum_steps=0):
        #Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:,1:, 0].contiguous().view(-1)
            lm_shifted = lm_logits[:,:-1,:].contiguous().view(-1,lm_logits.size(2))
            M         = M.view(-1, M.size(-1))
            lm_losses = self.lm_criterion(lm_shifted, x_shifted)
            lm_losses = lm_losses.view(X.size(0), X.size(-2) - 1)
            lm_losses = lm_losses * M[:, :-1]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, :-1], 1)

        if only_return_losses:
            return lm_losses
        train_loss = lm_losses.sum()
        train_loss.backward()
        if self.opt is not None and batch_num != None and (batch_num + 1) % accum_steps == 0:
            print('opt updating')
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

class SummarizationLossCompute2:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion,  opt=None):
        self.lm_criterion  = lm_criterion
        self.opt           = opt

    def __call__(self, lm_logits,lm_labels, encoder, batch_num=None, only_return_losses=False, accum_steps=0):
        #Language modeling loss
        if lm_logits is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()

            # Flatten the tokens
            loss = self.lm_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
        if only_return_losses:
            return loss.sum()
        train_loss = loss.sum()
        train_loss.backward()
        if self.opt is not None and batch_num != None and (batch_num + 1) % accum_steps == 0:
            #uncomment for debugging #print('opt updating')
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()

class ProcessLoss:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion,  opt=None):
        self.lm_criterion  = lm_criterion
        self.opt           = opt

    def __call__(self, loss, batch_num=None, only_return_losses=False, accum_steps=0):
        loss = loss.sum()
        loss.backward()
        if self.opt is not None and batch_num != None and (batch_num + 1) % accum_steps == 0:
            print('opt updating')
            self.opt.step()
            self.opt.zero_grad()
        return loss.item()

class SimLossCompute:
    "A Loss compute and train function for classification tasks."

    def __init__(self, lm_criterion,  opt=None):
        self.lm_criterion  = lm_criterion
        self.opt           = opt

    def __call__(self, Y, clf_logits, encoder, batch_num=None, only_return_losses=False,accum_steps=0):
        train_loss = self.lm_criterion(clf_logits,Y)
        if only_return_losses:
            return train_loss
        train_loss.backward()
        if self.opt is not None and batch_num != None and (batch_num + 1) % accum_steps == 0:
            print('opt updating')
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()
