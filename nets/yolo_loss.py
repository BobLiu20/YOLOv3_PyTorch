import torch
import torch.nn as nn
from torch.autograd import Variable

from common.utils import build_targets

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, image_dim):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.scaled_anchors = None
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = image_dim
        self.ignore_thres = 0.5
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.seen = 0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, targets=None):
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  self.image_dim / g_dim
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        self.seen += prediction.size(0)

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()

            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,
                                                                                                        targets.cpu().data,
                                                                                                        scaled_anchors,
                                                                                                        self.num_anchors,
                                                                                                        self.num_classes,
                                                                                                        g_dim,
                                                                                                        self.ignore_thres)


            nProposals = int((conf > 0.25).sum().item())

            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls[cls_mask == 1].type(FloatTensor), requires_grad=False)
            coord_mask = Variable(coord_mask.type(FloatTensor), requires_grad=False)
            conf_mask  = Variable(conf_mask.type(FloatTensor), requires_grad=False)

            loss_x = self.coord_scale * self.mse_loss(x[coord_mask == 1], tx[coord_mask == 1]) / 2
            loss_y = self.coord_scale * self.mse_loss(y[coord_mask == 1], ty[coord_mask == 1]) / 2
            loss_w = self.coord_scale * self.mse_loss(w[coord_mask == 1], tw[coord_mask == 1]) / 2
            loss_h = self.coord_scale * self.mse_loss(h[coord_mask == 1], th[coord_mask == 1]) / 2
            loss_conf = self.bce_loss(conf[conf_mask == 1], tconf[conf_mask == 1])
            loss_cls = self.class_scale * self.bce_loss(pred_cls[cls_mask == 1], tcls)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item()

        else:
            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data
