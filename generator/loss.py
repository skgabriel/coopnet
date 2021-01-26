import torch

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

