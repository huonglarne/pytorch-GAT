import torch

trg_index_broadcasted = torch.load('trg_index_broadcasted_cpu.pt').cuda()

neighborhood_sums = torch.rand(2708, 8, dtype=torch.float32).cuda()
exp_scores_per_edge = torch.rand(13264, 8, dtype=torch.float32).cuda()

neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)