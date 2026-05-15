import sys

import torch

def IID_loss_old(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  # p_i_j = compute_joint(x_out, x_tf_out)




  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
  su_temp1 = x_out.unsqueeze(2)
  su_temp2 = x_tf_out.unsqueeze(1)
  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k #这两个相乘有什么用？
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise 为什么要对称
  p_i_j = p_i_j / p_i_j.sum()  # normalise




  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
  #                           - torch.log(p_j) \
  #                           - torch.log(p_i))

  # loss_no_lamb = loss_no_lamb.sum()

  return loss



def IID_loss(N,x_out, x_tf_out, w_base,w_aux,bn,lamb=1.0, sample_weights=None, present_mask=None, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()

  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  if sample_weights is not None:
      p_i_j = p_i_j * sample_weights.view(-1, 1, 1)
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  p_i_j[(p_i_j < EPS).data] = EPS


  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))
 # w_base: known-known diagonal (dynamic, driven by source entropy)
  # w_aux:  unknown-unknown diagonal
  # bn:     unknown-unknown off-diagonal penalty (negative = discourage confusion)
  # 0.45:   known-known off-diagonal and cross-region weight
  weight_matrix = torch.full((k, k), 0.45, device=x_out.device)
  weight_matrix[N:,N:] = w_aux + bn
  weight_matrix[:N,:N] = 0.45
  for i in range(N):
       weight_matrix[i,i] = w_base
  for i in range(N,k):
    weight_matrix[i,i] = w_aux

  # DMI: suppress absent-absent region (classes not in this batch get negative weight)
  if present_mask is not None:
      n_present = present_mask.sum().item()
      if 0 < n_present < k:
          n_absent = k - n_present
          dmi_scale = np.log(n_present + 1) / np.log(n_absent + 1)
          absent_idx = torch.where(~present_mask)[0]
          weight_matrix[absent_idx[:, None], absent_idx[None, :]] *= -dmi_scale

  # ---------------------
  loss_all = p_i_j
  loss = (loss*weight_matrix).sum()

  return loss,loss_all


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j

