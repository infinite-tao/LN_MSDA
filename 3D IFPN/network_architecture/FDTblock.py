import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.autograd import Variable, Function


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        # print('pk', pk.shape)
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv3d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv3d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3], kernel.size()[4])
    px = x.view(1, -1, x.size()[2], x.size()[3], x.size()[4])
    # print('pk', pk.shape)
    # print('px', px.shape)
    po = F.conv3d(px, pk, **kwargs, groups=batch, padding=1)
    po = po.view(batch, -1, po.size()[2], po.size()[3], po.size()[4])
    # print('po', po.shape)
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv3d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3), x.size(4))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3), kernel.size(4))
        out = F.conv3d(x, kernel, **kwargs, groups=groups * batch, padding=1)
        out = out.view(batch, -1, out.size(2), out.size(3), out.size(4))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)

def add_conv3D(in_ch, out_ch, ksize, stride):
    """
    Add a conv3d / groupnorm / PReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv3d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('group_norm', nn.GroupNorm(32, out_ch))
    stage.add_module('leaky', nn.PReLU())
    return stage


class SAT_CAT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=8, **kwargs):
        super(SAT_CAT, self).__init__()
        self.region_num = region_num

        self.out_channels = out_channels

        self.conv1 = add_conv3D(64, 64, 3, (1, 2, 2))

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool3d((kernel_size, kernel_size, kernel_size)),
            nn.Conv3d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv3d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.conv_guide = nn.Conv3d(in_channels, region_num, kernel_size=kernel_size, **kwargs, padding=1)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

        self.o1 = nn.AdaptiveAvgPool3d(1)
        # self.o2 = nn.Linear(64, 32)

        self.o2 = nn.Sequential(nn.Conv3d(out_channels, 32, 1, bias=False),
                                 nn.GroupNorm(16, 32),
                                 nn.PReLU())
        self.fc2 = nn.Conv3d(32, out_channels * 2, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input1, input2):

        input1 = self.conv1(input1)

        kernel = self.conv_kernel(input1)

        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3), kernel.size(4))  # B x (r*in*out) x W X H

        output = self.corr(input2, kernel, **self.kwargs)  # B x (r*out) x W x H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3), output.size(4))  # B x r x out x W x H


        guide_feature = self.conv_guide(input1)

        output = self.asign_index(output, guide_feature)

        fusion1 = torch.add(output, input1)
        s = self.o1(fusion1)
        # weights = weights.view(weights.size(0), -1)
        z = self.o2(s)
        weights = self.fc2(z)

        weights = weights.reshape(input1.size(0), 2, self.out_channels, -1, 1)

        weights = self.softmax(weights)

        w1 = weights[:, 0:1, :, :, :]
        w1 = w1.reshape(w1.size(0), w1.size(2), w1.size(1), w1.size(3), w1.size(4))

        w2 = weights[:, 1:, :, :, :]
        w2 = w1.reshape(w2.size(0), w2.size(2), w2.size(1), w2.size(3), w2.size(4))
        final_output = torch.add(output*w1, input1*w2)

        return final_output

