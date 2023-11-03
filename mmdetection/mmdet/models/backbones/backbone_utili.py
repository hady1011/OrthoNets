import torch
from torch import Tensor

from model.FWT_MODULE import SCHMIDT_FWT_MODULE, BIG_SCHMIDT_FWT_MODULE, NEW_SCHMIDT_FWT_MODULE, \
    NEW_SCHMIDT_FWT_MODULE_BABY, SCHMIDT_FWT_MODULE_CORRECT_GRAM, GRAM_FCA


class WaveletAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: SCHMIDT_FWT_MODULE_CORRECT_GRAM, input: Tensor):
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)


class WaveletAttention_BIG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: BIG_SCHMIDT_FWT_MODULE, input: Tensor):
        # happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)


class WaveletAttention_NEW(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: NEW_SCHMIDT_FWT_MODULE, input: Tensor):
        # happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)


class WaveletAttention_NEW_BABY(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: NEW_SCHMIDT_FWT_MODULE_BABY, input: Tensor):
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)


class FCA_Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: GRAM_FCA, input: Tensor):
        # happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)