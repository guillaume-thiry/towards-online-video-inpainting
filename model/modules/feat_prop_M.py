"""
    BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment, CVPR 2022
"""
## We modify this file for Online-Memory inpainting as we now only inpaints the last frame and do not need to propagate the flow for all the frames anymore
#Besides, backward flow is not useful anymore for this last frame
import torch
import torch.nn as nn

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init

from model.modules.flow_comp import flow_warp


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super(BidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel,
                channel,
                3,
                padding=1,
                deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2*channel, channel, 1, 1, 0)

    #As only the last frame of the window needs to be inpainted now, the backward flow is not useful anymore
    def forward(self, x, flows_forward, feat_prop):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = x[:, 0, :, :, :]

        for module_name in ['forward_']:#Just forward

            feats[module_name] = []

            flows = flows_forward
            if flows==None:
                l = 0
            else:
                l = flows.size()[1]

            #Backward flow for this last frame is still calculated to reuse the current model
            #But because it is the last frame of the window, no information can be used (which is normal for this frame)
            back = [feats['spatial']] + [x.new_zeros(b, self.channel, h, w)]
            back = torch.cat(back, dim=1)
            back = self.backbone['backward_'](back)

            feat_current = feats['spatial']
            prop = x.new_zeros(b, self.channel, h, w)

            if l > 0:
                feat_prop_1 = feat_prop[-1]
                flow_n1 = flows[:, -1, :, :, :]
                cond_n1 = flow_warp(feat_prop_1, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_prop_2 = torch.zeros_like(feat_prop_1)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if l > 1:
                    feat_prop_2 = feat_prop[-2]
                    flow_n2 = flows[:, -2, :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_prop_2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                prop = torch.cat([feat_prop_1, feat_prop_2], dim=1)
                prop = self.deform_align[module_name](prop, cond, flow_n1, flow_n2)

            feat = [feat_current] + [back] + [prop]

            feat = torch.cat(feat, dim=1)
            prop = prop + self.backbone[module_name](feat)
            feat_prop.append(prop)

        align_feats = torch.cat([back,prop], dim=1)
        output = self.fusion(align_feats)

        return output + x, feat_prop #Outputs updated feat_prop for the memory
