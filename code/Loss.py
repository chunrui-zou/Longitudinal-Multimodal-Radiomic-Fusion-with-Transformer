import torch
import torch.nn as nn



# Custom loss function
def Multilabel_Loss(targets, predictions,
		    sample_weight=None,pos_weight=None,
                    label_weight = None,weight=None):

    '''Get the total loss for multi-label classification.

    Inputs:
        targets: tensor of shape (samples, n_labs)
        predictions: tensor of shape (samples, n_labs)
        sample_weigtht: w(n) as a way to balance the weight between different samples. This weight can be attribute to the weights of certain sequence length.
        pos_weight: p(n_labs), as a way to balance the loss between positive and negative labels inside each label.
	label_weight: p(n_labs), instead of inside each label, the weights balance losses between labels.
                             
    Outputs:
        loss: the average loss across several labels
        lss_labs: the loss separate for each label.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        
    '''
    loss_fn = nn.BCEWithLogitsLoss(weight=sample_weight,reduction="none",pos_weight=pos_weight)
    loss_individual = loss_fn(predictions.float(), targets.float()) # n_samples,n_labs
    loss_labs = torch.mean(loss_individual,dim=0) # for monitoring losses for each label.
    if label_weight is not None:
        loss = torch.mean(loss_labs*label_weight)
    else:
        loss = torch.mean(loss_labs)
    return loss, loss_labs
