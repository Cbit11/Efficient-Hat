import torch 
import torch.nn as nn 

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class point_wise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.point_wise= nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.point_wise(x)
    
class depth_wise(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depth_wise= nn.Conv2d( in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
    def forward(self, x):
        return self.depth_wise(x)
    
class mobile_sub(nn.Module):
    def __init__(self, in_chans, embed_dims):
        super(mobile_sub, self).__init__()
        self.in_chans= in_chans
        self.embed_dims = embed_dims
        self.act= nn.ReLU(inplace=True)
        self.point_wise= point_wise(in_channels=in_chans, out_channels=embed_dims)
        self.depth_wise= depth_wise(in_channels=embed_dims)
        self.point_wise_2= point_wise(in_channels=embed_dims, out_channels=embed_dims)
    def forward(self, x):
        x= self.act(self.point_wise(x))
        x= self.act(self.depth_wise(x))
        return self.point_wise_2(x)


class mobile2former(nn.Module):
    def __init__(self, embed_dims, num_heads,qkv_bias=True, qk_scale=None, attn_drop=0.):
        super(mobile2former, self).__init__()
        self.embed_dims= embed_dims
        self.num_heads= num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.wo = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
    def forward(self, x):
        B_, N, C= x.shape
        k, v= x,x
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k= k.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v= v.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = self.softmax((q @ k.transpose(-2, -1)))
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x= self.wo(x)+ x
        return x

class former2mobile(nn.Module):
    def __init__(self, embed_dims, num_heads,qkv_bias=True, qk_scale=None, attn_drop=0.):
        super(former2mobile, self).__init__()
        self.embed_dims= embed_dims
        self.num_heads= num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v= nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
    def forward(self, Q,K,V):
        B_, N, C= Q.shape
        q= Q
        k = self.k(K).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v= self.v(V).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q= q.reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = self.softmax((q @ k.transpose(-2, -1)))
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads,qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0.):
        super(MultiHeadAttention, self).__init__()
        self.embed_dims= embed_dims
        self.num_heads= num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B_, N, C= x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mobile_Former_block(nn.Module):
    def __init__(self, in_chans, embed_dims, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0.):
        super(Mobile_Former_block,self).__init__()
        self.mobile_net= mobile_sub(embed_dims, embed_dims)
        self.mobile2former= mobile2former(embed_dims, num_heads)
        self.former2mobile= former2mobile(embed_dims, num_heads)
        self.mha = MultiHeadAttention(embed_dims, num_heads)
        self.act= nn.ReLU(inplace=True)
    def forward(self, x, x_size):
         
        # x: B, C, H, W
        H, W= x_size
        B, _,C = x.shape 
        x_mobile = self.mobile_net(x.view(B,C,H, W)).contiguous().view(B, H*W,C).permute(0,1,2)
        x_mobile2former= self.mobile2former(x)
        x_mobile2former= self.mha(x_mobile2former)
        
        out= self.former2mobile(x_mobile, x_mobile2former, x_mobile2former)+ x_mobile
        return out

# x= torch.randn(4, 64*64, 96).to('cuda')
# sub = Mobile_Former_block(96, 96, 8).to('cuda')
# y=sub(x, (64, 64))
# print(y.shape)