import torch
from torch import Tensor

from model.transforms import GramSchmidtTransform

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)