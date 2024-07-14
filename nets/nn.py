import torch


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, r, k, s, fused=True):
        super().__init__()
        self.add = s == 1 and in_ch == out_ch

        if fused:
            modules = [Conv(in_ch, r * in_ch, torch.nn.ReLU(), k, s, (k - 1) // 2),
                       Conv(r * in_ch, out_ch, torch.nn.Identity()) if r != 1 else torch.nn.Identity()]
        else:
            modules = [Conv(in_ch, r * in_ch, torch.nn.ReLU()) if r != 1 else torch.nn.Identity(),
                       Conv(r * in_ch, r * in_ch, torch.nn.ReLU(), k, s, (k - 1) // 2, g=r * in_ch),
                       Conv(r * in_ch, out_ch, torch.nn.Identity())]

        self.res = torch.nn.Sequential(*modules)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class Backbone(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(), k=3, s=2, p=1))
        self.p1.append(Residual(filters[1], filters[1], r=1, k=3, s=1))
        # p2/4
        self.p2.append(Residual(filters[1], filters[2], r=3, k=3, s=2))
        self.p2.append(Residual(filters[2], filters[2], r=3, k=3, s=1))
        # p3/8
        self.p3.append(Residual(filters[2], filters[3], r=3, k=3, s=2))
        self.p3.append(Residual(filters[3], filters[3], r=3, k=3, s=1))
        # p4/16
        self.p4.append(Residual(filters[3], filters[4], r=6, k=3, s=2, fused=False))
        self.p4.append(Residual(filters[4], filters[4], r=3, k=3, s=1, fused=False))
        self.p4.append(Residual(filters[4], filters[4], r=3, k=3, s=1, fused=False))
        self.p4.append(Residual(filters[4], filters[4], r=3, k=3, s=1, fused=False))
        self.p4.append(Residual(filters[4], filters[4], r=6, k=3, s=1, fused=False))
        self.p4.append(Residual(filters[4], filters[4], r=6, k=3, s=1, fused=False))
        # p5/32
        self.p5.append(Residual(filters[4], filters[5], r=6, k=5, s=2, fused=False))
        self.p5.append(Residual(filters[5], filters[5], r=6, k=5, s=1, fused=False))
        self.p5.append(Residual(filters[5], filters[5], r=6, k=5, s=1, fused=False))
        self.p5.append(Residual(filters[5], filters[5], r=6, k=5, s=1, fused=False))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5


# Feature Fusion Module
class FFM(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv(ch, ch, torch.nn.ReLU(True), k=3, p=1)
        self.conv2 = Conv(ch, ch, torch.nn.ReLU(True), k=3, p=1)

    def forward(self, x1, x2):
        y = x1 * x2
        return self.conv2(self.conv1(y))


# Cross Aggregation Module
class CAM(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv(ch, ch, torch.nn.ReLU(True), k=3, s=2, p=1)
        self.conv2 = Conv(ch, ch, torch.nn.ReLU(True), k=3, s=1, p=1)
        self.conv3 = Conv(ch, ch, torch.nn.ReLU(True), k=3, s=1, p=1)
        self.conv4 = FFM(ch)

    def forward(self, x1, x2):
        y1 = self.conv1(x2)
        y2 = torch.nn.functional.interpolate(x1,
                                             size=x2.size()[2:],
                                             mode='bilinear', align_corners=True)
        a = self.conv2(x2 * y2)
        b = self.conv3(y1 * x1)
        b = torch.nn.functional.interpolate(b,
                                            size=x2.size()[2:],
                                            mode='bilinear', align_corners=True)
        return self.conv4(a, b)


# Spatial Attention Module
class SAM(torch.nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.conv2 = Conv(in_chan, out_chan, torch.nn.ReLU(), k=3, p=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        y = torch.cat(tensors=[avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv1(y))

        return self.conv2(torch.mul(x, y))


# Boundary Refinement Module
class BRM(torch.nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.conv2 = Conv(channel, channel, torch.nn.ReLU(), k=3, p=1)
        self.conv3 = Conv(channel, channel, torch.nn.ReLU(), k=3, p=1)

    def forward(self, x1, x2):
        x = x1 + x2
        y = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        y = torch.sigmoid(self.conv1(y))
        y = torch.mul(x, y) + x
        y = self.conv2(y)
        y = self.conv3(y)
        return y


class CTDNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        filters = [3, 16, 24, 40, 80, 160]
        self.backbone = Backbone(filters)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = Conv(filters[5], filters[3], torch.nn.Identity())
        self.conv2 = Conv(filters[5], filters[3], torch.nn.Identity())
        self.conv3 = Conv(filters[4], filters[3], torch.nn.Identity())
        self.conv4 = SAM(filters[3], filters[3])
        self.conv5 = Conv(filters[2], filters[3], torch.nn.Identity())

        self.fuse1 = FFM(filters[3])
        self.fuse2 = FFM(filters[3])
        self.fuse3 = CAM(filters[3])
        self.fuse4 = FFM(filters[3])
        self.fuse5 = BRM(filters[3])

        self.cls1 = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)
        self.cls2 = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)
        self.cls3 = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)
        self.cls4 = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)
        self.cls5 = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)
        self.edge = torch.nn.Conv2d(filters[3], out_channels=1, kernel_size=1)

    def forward(self, x):
        shape = x.size()[2:]
        p2, p3, p4, p5 = self.backbone(x)

        cls5 = self.conv1(self.avg_pool(p5))
        cls5 = torch.nn.functional.interpolate(cls5,
                                               size=p5.size()[2:],
                                               mode='bilinear', align_corners=True)  # 1/32
        cls4 = self.fuse1(cls5, self.conv2(p5))  # 1/32
        cls4 = torch.nn.functional.interpolate(cls4,
                                               size=p4.size()[2:],
                                               mode='bilinear', align_corners=True)  # 1/16

        cls3 = self.fuse2(cls4, self.conv3(p4))  # 1/16

        cls2 = self.fuse3(cls3, self.conv4(p3))  # 1/8
        cls2 = torch.nn.functional.interpolate(cls2,
                                               size=p2.size()[2:],
                                               mode='bilinear', align_corners=True)  # 1/4

        cls1 = torch.nn.functional.interpolate(cls4,
                                               size=p2.size()[2:],
                                               mode='bilinear', align_corners=True)  # 1/4
        edge = self.fuse4(self.conv5(p2), cls1)  # 1/4
        cls1 = self.fuse5(cls2, edge)  # 1/4

        if self.training:
            cls1 = torch.nn.functional.interpolate(self.cls1(cls1),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)
            cls2 = torch.nn.functional.interpolate(self.cls2(cls2),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)
            cls3 = torch.nn.functional.interpolate(self.cls3(cls3),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)
            cls4 = torch.nn.functional.interpolate(self.cls4(cls4),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)
            cls5 = torch.nn.functional.interpolate(self.cls5(cls5),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)
            edge = torch.nn.functional.interpolate(self.edge(edge),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)
            return [cls1, cls2, cls3, cls4, cls5, edge]
        else:
            return torch.nn.functional.interpolate(self.cls1(cls1),
                                                   size=shape,
                                                   mode='bilinear', align_corners=True)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
