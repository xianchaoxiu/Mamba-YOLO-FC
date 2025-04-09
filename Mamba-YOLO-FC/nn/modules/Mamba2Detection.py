import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class Mamba2Detection(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,  # 输入图像的通道数，默认RGB图像
                 n_layer: int = 24,     # Mamba-2层数（针对图像处理可以调整）
                 d_state: int = 128,    # 状态维度（N）
                 d_conv: int = 4,       # 卷积核大小（D）
                 expand: int = 2,       # 扩展因子（E）
                 headdim: int = 64,     # 头部维度（P）
                 chunk_size: int = 64,  # 矩阵分块大小（Q）
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
       

        self.d_inner = expand * in_channels  # 使用输入图像的通道数来计算
        assert self.d_inner % self.headdim == 0, "self.d_inner must be divisible by self.headdim"
        self.nheads = self.d_inner // self.headdim

        # 调整输入投影层，适应图像特征
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Conv2d(in_channels, d_in_proj, kernel_size=3, stride=1, padding=1, bias=False)

        # 卷积层，用于提取图像特征
        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv2d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1)
        
        # 偏置和参数
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        
     
        
        # 输出投影层，用于目标检测的边界框和类别预测
        self.box_head = nn.Conv2d(self.d_inner, 4, kernel_size=1)  # 边界框回归（4个坐标值）
        self.cls_head = nn.Conv2d(self.d_inner, self.num_classes, kernel_size=1)  # 类别预测

    def forward(self, x: torch.Tensor):
        # A_log初始化，输出A (注意力缩放因子)
        A = -torch.exp(self.A_log)  # (nheads,)
        
        # 输入投影层，将图像特征映射到更高维度
        zxbcdt = self.in_proj(x)  # (batch, channels, height, width)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)

        # 通过卷积层提取图像特征
        xBC = F.silu(self.conv1d(xBC))  # (batch, channels, height, width)
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=1)

        # 调整维度以适应头部（多头注意力）
        _b, _c, _h, _w = x.shape
        _h_dim = _c // self.headdim
        x = x.reshape(_b, _h, _w, _h_dim, self.headdim)

        x = self.out_proj(x)

        return x


def segsum(self, x: Tensor) -> Tensor:
    """
    Function to compute cumulative sum over segments, ensuring values outside
    the lower triangular part of the segment matrix are masked.
    
    Args:
    - x (Tensor): Input tensor with shape (B, C, L, H, P), where L is the sequence length.

    Returns:
    - Tensor: Segmented cumulative sum tensor with shape (B, C, L, H, P)
    """
    T = x.size(-2)  # Assume the length is the second to last dimension, L.
    device = x.device
    
    # Expand x to be able to operate with a triangular mask (B, C, L, H, P, T)
    x = x[..., None].repeat(1, 1, 1, 1, 1, T)
    
    # Create lower triangular mask of size (T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    
    # Apply mask and compute cumulative sum along the second-to-last dimension
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    
    # Apply mask again to enforce the condition
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    
    return x_segsum


def ssd(self, x: Tensor, A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    """
    Function for computing the state-space model recurrence with chunking.
    
    Args:
    - x (Tensor): Input tensor with shape (B, L, H, P), where L is the sequence length.
    - A (Tensor): Matrix A with shape (B, L, K) representing the transition matrix.
    - B (Tensor): Matrix B with shape (B, K, H, P) used for state update.
    - C (Tensor): Matrix C with shape (B, L, H, P) used for output computation.
    
    Returns:
    - Tensor: Output tensor with shape (B, C*L, H, P)
    """
    chunk_size = self.chunk_size

    # Reshape tensors into chunks, making sure dimensions are divisible by chunk_size
    x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3])
    B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3])
    C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3])
    A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
    
    # Transpose A for easier matrix operations
    A = A.permute(0, 3, 1, 2)
    
    # Compute cumulative sum for A
    A_cumsum = torch.cumsum(A, dim=-1)
    
    # Step 1: Compute diagonal blocks (intra-chunk interactions)
    L = torch.exp(self.segsum(A))  # Apply segsum on A to get the lower-triangular cumulative sum
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    
    # Step 2: Compute state decay for intra-chunk updates
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    
    # Step 3: Handle inter-chunk state recurrence
    initial_states = torch.zeros_like(states[:, :1])  # Add initial state as zeros
    states = torch.cat([initial_states, states], dim=1)  # Concatenate initial state
    
    decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states = new_states[:, :-1]
    
    # Step 4: Compute final output after the inter-chunk recurrence
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    
    # Combine diagonal and off-diagonal contributions
    Y = Y_diag + Y_off