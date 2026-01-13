import torch
import torch.nn as nn

class DeltaFreeUSchedule(nn.Module):
    def __init__(self, K=25, T=1000, max_pct=0.2,
                 base_b1=1.4, base_b2=1.6, base_s1=0.9, base_s2=0.2):
        super().__init__()
        self.K, self.T, self.max_pct = K, T, max_pct
        self.base_b1, self.base_b2 = base_b1, base_b2
        self.base_s1, self.base_s2 = base_s1, base_s2

        self.db1 = nn.Parameter(torch.zeros(K))
        self.db2 = nn.Parameter(torch.zeros(K))
        self.ds1 = nn.Parameter(torch.zeros(K))
        self.ds2 = nn.Parameter(torch.zeros(K))

    def _idx_from_t(self, t):
        return torch.clamp((t.float() / (self.T - 1) * (self.K - 1)).long(), 0, self.K - 1)

    def forward(self, t):
        idx = self._idx_from_t(t)

        def delta(x):
            return self.max_pct * torch.tanh(x)  

        b1 = self.base_b1 * (1.0 + delta(self.db1[idx]))
        b2 = self.base_b2 * (1.0 + delta(self.db2[idx]))
        s1 = self.base_s1 * (1.0 + delta(self.ds1[idx]))
        s2 = self.base_s2 * (1.0 + delta(self.ds2[idx]))

        s1 = torch.clamp(s1, 0.05, 1.0)
        s2 = torch.clamp(s2, 0.05, 1.0)
        return b1, b2, s1, s2