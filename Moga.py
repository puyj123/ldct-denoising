import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


'''
class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=1,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            65, embed_dims, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        #print(x.shape)
        return x, out_size
'''
class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=1,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        #self.act = nn.GELU()

    def forward(self, x):
        x = self.projection(x)
        #x = self.act(x)
        #out_size = (x.shape[2], x.shape[3])
        return x

class SFEM(nn.Module):
    """An implementation of Shallow feature extraction module.

    Args:
        in_features (int): The feature dimension.
        out_channels (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 act_type='GELU',
                 norm_type='BN'):
        super(SFEM, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            #build_norm_layer(norm_type, out_channels // 2),
            build_act_layer(act_type),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2),
            #build_norm_layer(norm_type, out_channels),
        )

    def forward(self, x):
        x = self.projection(x)
        # out_size = (x.shape[2], x.shape[3])
        # print(x.shape)
        return x


class SFDM(nn.Module):
    """An implementation of Shallow feature extraction module.

    Args:
        in_features (int): The feature dimension.
        out_channels (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 act_type='GELU',
                 norm_type='BN'):
        super(SFDM, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            #build_norm_layer(norm_type, in_channels // 2),
            build_act_layer(act_type),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2),
            #build_norm_layer(norm_type, out_channels),
        )

    def forward(self, x):
        x = self.projection(x)
        # out_size = (x.shape[2], x.shape[3])
        # print(x.shape)
        return x

class Upsampel(nn.Module):
    """An implementation of Shallow feature extraction module.

    Args:
        in_features (int): The feature dimension.
        out_channels (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 act_type='GELU',
                 norm_type='BN'):
        super(Upsampel, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            #build_norm_layer(norm_type, out_channels),
            #build_act_layer(act_type),
        )

    def forward(self, x):
        x = self.projection(x)
        # out_size = (x.shape[2], x.shape[3])
        # print(x.shape)
        return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, ],
                 channel_split=[1, 1, 3, ],
                 ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        #print(self.out_channels)
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        #x = self.layer_scale_1 * self.attn(x)
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        #x = self.layer_scale_2 * self.mlp(x)
        x = identity + self.drop_path(x)
        return x


class Moga(nn.Module):
    def __init__(self, in_channels=1,
                 embed_dims=32,
                 sobel_ch=32,
                 stem_norm_type='BN',
                 attn_force_fp32=False,
                 **kwargs):
        super(Moga, self).__init__()

        self.embed_dims = 32
        self.sobel_ch = 32
        self.in_channels = 1
        self.conv_sobel = SobelConv2d(self.in_channels, self.sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
        #self.SFEM1 = torch.nn(self.in_channels+self.sobel_ch+self.embed_dims, self.embed_dims)
        self.SFEM = SFEM(self.in_channels, self.embed_dims)

        self.ffn_ratios = [8, 8, 4, 4]
        # self.num_stages = len(self.depths)
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_type == 'LN'
        self.block11 = MogaBlock(embed_dims=self.embed_dims)
        self.block21 = MogaBlock(embed_dims=self.embed_dims*2)
        self.block31 = MogaBlock(embed_dims=self.embed_dims*4)
        self.block41 = MogaBlock(embed_dims=self.embed_dims*2)
        #self.block51 = MogaBlock(embed_dims=self.embed_dims)

        # norm = build_norm_layer(stem_norm_type, self.embed_dims[i])
        self.SFDM = SFDM(in_channels=self.embed_dims, out_channels=self.in_channels)
        self.Upsame_1 = Upsampel(in_channels=self.embed_dims * 4, out_channels=self.embed_dims*2)
        self.Upsame_2 = Upsampel(in_channels=self.embed_dims * 2, out_channels=self.embed_dims)
        self.Downsampe_1 = ConvPatchEmbed(in_channels=self.embed_dims, embed_dims=self.embed_dims*2)
        #self.Downsampe_s = ConvPatchEmbed(in_channels=self.sobel_ch, embed_dims=self.sobel_ch*2)
        self.Downsampe_2 = ConvPatchEmbed(in_channels=self.embed_dims*2, embed_dims=self.embed_dims*4)

        #self.SFEM3 = ConvPatchEmbed(in_channels=self.embed_dims*4+self.sobel_ch+self.in_channels, embed_dims=self.embed_dims*6)
        #self.SFEM4 = ConvPatchEmbed(in_channels=self.embed_dims*6+self.sobel_ch+self.in_channels, embed_dims=self.embed_dims*10)
        self.norm1 = build_norm_layer(norm_type='BN', embed_dims=self.embed_dims + self.sobel_ch + self.in_channels)
        self.norm2 = build_norm_layer(norm_type='BN', embed_dims=self.embed_dims*2 + self.sobel_ch + self.in_channels)

        self.conv_m1 = nn.Conv2d(in_channels=self.embed_dims+self.sobel_ch+self.in_channels, out_channels=self.embed_dims, kernel_size=1)
        self.conv_m2 = nn.Conv2d(in_channels=self.embed_dims*2+self.sobel_ch+self.in_channels, out_channels=self.embed_dims*2, kernel_size=1
                               )
        self.conv_m3 = nn.Conv2d(in_channels=self.embed_dims * 2 + self.sobel_ch + self.in_channels, out_channels=self.embed_dims * 2,
                                 kernel_size=1
                                 )
        self.conv_m4 = nn.Conv2d(in_channels=self.embed_dims+self.sobel_ch+self.in_channels, out_channels=self.embed_dims, kernel_size=1)

        #self.conv2 = nn.Conv2d(in_channels=self.embed_dims*2+self.sobel_ch+self.in_channels, out_channels=self.embed_dims*4, kernel_size=1)
        #self.conv3 = nn.Conv2d(in_channels=self.embed_dims*4+self.sobel_ch+self.in_channels, out_channels=self.embed_dims*6, kernel_size=1)
        #self.conv4 = nn.Conv2d(in_channels=self.embed_dims*6+self.sobel_ch+self.in_channels, out_channels=self.embed_dims*10, kernel_size=1)

        self.relu = nn.GELU()

    def forward(self, x):
        #print(x.shape)
        out_0 = self.conv_sobel(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        out_1 = self.SFEM(x)
        # out_1 = torch.cat(out_1, out_e)
        #out_1 = torch.cat((out_1, out_0), dim=-3)

        #out_2 = self.SFEM2(out_1)
        #out_2 = self.conv2(out_1)
        #out_2 = self.relu(out_2)
        out_1 = self.block11(out_1)
        out_1 = self.block11(out_1)
        out_1 = self.block11(out_1)
        out_1 = self.block11(out_1)
        out_1 = torch.cat((out_1, out_0), dim=-3)
        out_1 = self.norm1(out_1)
        #out_2 = self.block2(out_2)
        #out_2 = self.relu(out_2)
        #out_2 = self.block(out_2)
        #out_2 = self.relu(out_2)
        #out_2 = torch.cat((out_2, out_0), dim=-3)

        out_1 = self.conv_m1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.Downsampe_1(out_1)
        #out_1 = self.relu(out_1)

       #out_3 = self.conv3(out_2)
        #out_3 = self.SFEM3(out_2)
        #out_3 = self.relu(out_3)
        #out_3 = self.conv2d(out_2)
        #out_3 = self.block(out_3)
        #out_3 = self.relu(out_3)
        out_2 = self.block21(out_1)
        out_2 = self.block21(out_2)
        out_2 = self.block21(out_2)
        out_2 = self.block21(out_2)
        #out_2 = self.block2(out_2)
        #out_2 = self.block2(out_2)
        #out_0d = self.Downsampe_s(out_0)
        out_2 = torch.cat((out_2, out_0), dim=-3)
        out_2 = self.norm2(out_2)
        out_2 = self.conv_m2(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.Downsampe_2(out_2)
        #out_2 = self.relu(out_2)


        # = self.relu(out_3)
        #out_4 = self.SFEM4(out_3)
        out_3 = self.block31(out_2)
        out_3 = self.block31(out_3)
        out_3 = self.block31(out_3)
        out_3 = self.block31(out_3)
        #out_3 = self.block3(out_3)
        #out_3 = self.relu(out_3)


        "up-sampeling"
        out_4 = self.Upsame_1(out_3)
        #out_4 = self.relu(out_4)
        out_4 = torch.cat((out_4, out_0), dim=-3)
        #out_4 = self.norm2(out_4)
        #out_4 = self.relu(out_4)
        out_4 = self.conv_m3(out_4)
        out_4 = self.block41(out_4)
        out_4 = self.block41(out_4)
        out_4 = self.block41(out_4)
        out_4 = self.block41(out_4)
        #out_4 = self.block4(out_4)
        #out_4 = self.block4(out_4)
        out_4 = self.Upsame_2(out_4)
        #out_4 = self.relu(out_4)

        out_5 = torch.cat((out_4, out_0), dim=-3)
        #out_5 = self.norm1(out_5)
        #out_5 = self.relu(out_5)
        out_5 = self.conv_m4(out_5)
        #out_5 = self.block51(out_5)
        #out_5 = self.block51(out_5)
        #out_5 = self.block51(out_5)
        #out_5 = self.block51(out_5)
        #out_5 = self.block5(out_5)
        #out_5 = self.block5(out_5)

        out = self.SFDM(out_5)
        out = x - out
        return out
        #out_5 = self.relu(out_5)
        #out_5 = self.block(out_5)
        #out_5 = self.relu(out_5)
        #out_5 = torch.cat((out_5, out_0), dim=-3)
'''
        out_6 = self.conv(out_5)
        #out_6 = self.relu(out_6)
        out_6 = self.block(out_6)
        out_6 = self.block(out_6)
        #out_6 = self.relu(out_6)
        #out_6 = self.block(out_6)
       # out_6 = self.relu(out_6)
        out_6 = torch.cat((out_6, out_0), dim=-3)

        out_7 = self.conv(out_6)
        #out_7 = self.relu(out_7)
        out_7 = self.block(out_7)
        out_7 = self.block(out_7)
        #out_7 = self.relu(out_7)
        #out_7 = self.block(out_7)
        #out_7 = self.relu(out_7)
        out_7 = torch.cat((out_7, out_0), dim=-3)

        out_8 = self.conv(out_7)
        #out_8 = self.relu(out_8)
        out_8 = self.block(out_8)
        out_8 = self.block(out_8)
        #out_8 = self.relu(out_8)
        #out_8 = self.block(out_8)
        #out_8 = self.relu(out_8)
        out_8 = torch.cat((out_8, out_0), dim=-3)

        out_9 = self.conv(out_8)
        #out_9 = self.relu(out_9)
        out_9 = self.block(out_9)
        out_9 = self.block(out_9)
        #out_9 = self.relu(out_9)
        #out_9 = self.block(out_9)
        #out_9 = self.relu(out_9)
        '''
class EDCNN(nn.Module):

    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32):
        super(EDCNN, self).__init__()

        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_p1 = nn.Conv2d(in_ch + sobel_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p5 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f5 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p6 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f6 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p7 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f7 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p8 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f8 = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_0 = self.conv_sobel(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        out_1 = self.relu(self.conv_p1(out_0))
        out_1 = self.relu(self.conv_f1(out_1))
        out_1 = torch.cat((out_0, out_1), dim=-3)

        out_2 = self.relu(self.conv_p2(out_1))
        out_2 = self.relu(self.conv_f2(out_2))
        out_2 = torch.cat((out_0, out_2), dim=-3)

        out_3 = self.relu(self.conv_p3(out_2))
        out_3 = self.relu(self.conv_f3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=-3)

        out_4 = self.relu(self.conv_p4(out_3))
        out_4 = self.relu(self.conv_f4(out_4))
        out_4 = torch.cat((out_0, out_4), dim=-3)

        out_5 = self.relu(self.conv_p5(out_4))
        out_5 = self.relu(self.conv_f5(out_5))
        out_5 = torch.cat((out_0, out_5), dim=-3)

        out_6 = self.relu(self.conv_p6(out_5))
        out_6 = self.relu(self.conv_f6(out_6))
        out_6 = torch.cat((out_0, out_6), dim=-3)

        out_7 = self.relu(self.conv_p7(out_6))
        out_7 = self.relu(self.conv_f7(out_7))
        out_7 = torch.cat((out_0, out_7), dim=-3)

        out_8 = self.relu(self.conv_p8(out_7))
        out_8 = self.conv_f8(out_8)

        out = self.relu(x + out_8)

        return out
