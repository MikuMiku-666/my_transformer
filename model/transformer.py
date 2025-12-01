import torch
from torch import nn 
from torch.nn import functional as F
from torch import Tensor

class transformer_layer(nn.Module):
    def __init__(self, maxPoints : int, in_dim : int, out_dim : int, share_dim = 16) -> None:
        super().__init__()
        self.maxPoints = maxPoints
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim = out_dim // 1
        self.share_dim = share_dim

        self.linear_q = nn.Linear(in_dim, mid_dim)
        self.linear_k = nn.Linear(in_dim, mid_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)
        
        self.mlp_pos = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3),   
                                      nn.ReLU(inplace=True), nn.Linear(3, mid_dim)) #pos function
        self.mlp_w = nn.Sequential(nn.BatchNorm1d(mid_dim), nn.ReLU(inplace=True), 
                                   nn.Linear(mid_dim, mid_dim), nn.ReLU(inplace=True), 
                                   nn.Linear(mid_dim, out_dim)) # No need to group as our point cloud is small
        # self.soft_max = nn.Softmax(-1)
        
    def forward(self, pos : Tensor, c_features : Tensor) -> Tensor:
        '''pos: [Batch, N, 3] 
        c_features: [Batch, N, c], where c must matches the in_dim of transformer_layer
        Return: [Batch, N, out_dim]
        '''

        B = pos.shape[0]
        N = pos.shape[1]

        q = self.linear_q(c_features)
        k = self.linear_k(c_features)
        v = self.linear_v(c_features)   # [Batch, N, out_dim]

        pos_i = pos.unsqueeze(2)          # [Batch, N, 1, 3]
        pos_j = pos.unsqueeze(1)          # [Batch, 1, N, 3]
        rel_pos = pos_i - pos_j           # [Batch, N, N, 3], rel_pos[i, j] = p_i - p_j

        rel_pos_flat = rel_pos.reshape(B * N * N, 3)        # [B*N*N, 3]
        delta_flat   = self.mlp_pos(rel_pos_flat)    # [B*N*N, mid_dim]
        delta        = delta_flat.view(B, N, N, self.mid_dim)  # [B, N, N, mid_dim]
        # delta: pos function's result

        q_i = q.unsqueeze(2)             # [B, N, 1, C]
        k_j = k.unsqueeze(1)             # [B, 1, N, C]
        qk_delta = q_i - k_j + delta     # [B, N, N, C]

        qk_delta_flat = qk_delta.reshape(B * N * N, self.mid_dim)  # [B*N*N, C]
        w_flat = self.mlp_w(qk_delta_flat)                         # [B*N*N, C]
        w = w_flat.view(B, N, N, self.out_dim)                     # [B, N, N, C]

        attention_score = F.softmax(w, dim=2)                      # [B, N, N, C]


        v_flatten = v.unsqueeze(1).expand(-1, N, -1, -1)
        out = (attention_score * (v_flatten + delta)).sum(2)  # [B, N, out_dim(c)]

        return out
    

class Point_transformer(nn.Module):
    def __init__(self, in_dim, out_points = 19) -> None:
        '''out_points: 19 initially'''
        super().__init__()

        self.in_dim = in_dim
        self.out_points = out_points

        self.tformer_layer = transformer_layer(32, in_dim, in_dim)
        self.tformer_layer2 = transformer_layer(32, in_dim, in_dim)
        self.tformer_layer3 = transformer_layer(32, in_dim, in_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), 
            nn.BatchNorm1d(in_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_dim, out_points * 3)
        )

    def forward(self, pos, c_features)  -> Tensor: 
        '''pos: [Batch, N, 3] 
        c_features: [Batch, N, c], where c must matches the in_dim of transformer_layer
        Return: [Batch, out_points, 3]   
        '''
        B = pos.shape[0]
        x = self.tformer_layer(pos, c_features)  # [B, N, c]
        x = x + self.tformer_layer2(pos, x)
        x = x + self.tformer_layer3(pos, x)
        x = x.max(dim=1).values    
        x = self.mlp.forward(x)     # [B, out_points * 3]
        x = x.view(B, self.out_points, 3)
        return x


    

        








        


