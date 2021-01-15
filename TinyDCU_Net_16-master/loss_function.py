import torch

EPSILON = 1e-10


def mse_loss_stage_1(esti_list, label):
    masked_esti_0 = esti_list[0]
    masked_label = label.squeeze()
    loss1 = ((masked_esti_0 - masked_label) ** 2).mean()
    return loss1 + EPSILON

def mse_loss_stage_2(esti_list, label):
    masked_esti_0 = esti_list[0]
    masked_esti_1 = esti_list[1]
    masked_label = label.squeeze()
    loss1 = ((masked_esti_0 - masked_label) ** 2).mean()
    loss2 = ((masked_esti_1 - masked_label) ** 2).mean()
    return (loss1 + loss2) / 2 + EPSILON

def mse_loss_stage_3(esti_list, label):
    masked_esti_0 = esti_list[0]
    masked_esti_1 = esti_list[1]
    masked_esti_2 = esti_list[2]
    masked_label = label.squeeze()
    loss1 = ((masked_esti_0 - masked_label) ** 2).mean()
    loss2 = ((masked_esti_1 - masked_label) ** 2).mean()
    loss3 = ((masked_esti_2 - masked_label) ** 2).mean()
    return (loss1 + loss2 + loss3) / 3 + EPSILON
