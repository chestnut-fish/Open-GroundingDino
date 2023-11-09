import torch


logits = torch.randint(low=0, high=100, size=(4, 2, 2))
boxes = torch.randn(size=(4, 2, 4))





logits_max = logits.max(dim=-1).values
max_logits, max_logits_indices = torch.max(logits_max, dim=1)
selected_boxes = torch.gather(boxes, 1, max_logits_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4))



# logits_filt = logits_filt[filt_mask]  # bs, num_filt, 256
# print(logits_filt.shape)

# boxes_filt = boxes_filt[filt_mask]  # bs, num_filt, 4

# print(boxes_filt.shape)
