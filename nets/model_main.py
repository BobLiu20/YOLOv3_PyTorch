import torch
import torch.nn as nn

from .backbone import backbone_fn


class ModelMain(nn.Module):
    def __init__(self, config, is_training=True):
        super(ModelMain, self).__init__()
        self.config = config
        self.training = is_training
        self.model_params = config["model_params"]
        #  backbone
        _backbone_fn = backbone_fn[self.model_params["backbone_name"]]
        self.backbone = _backbone_fn()
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1])
        #  embedding1
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        #  embedding2
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(_out),
            nn.LeakyReLU(0.1),
        )

    def _make_embedding(self, filters_list, in_filters):
        return nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], 255, 1)])

    def forward(self, x):
        def _branch(_embedding, _in):
            for i in range(len(_embedding)):
                _in = _embedding[i](_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        return out0, out1, out2

if __name__ == "__main__":
    config = {"model_params": {"backbone_name": "darknet_53"}}
    m = ModelMain(config)
    x = torch.randn(1, 3, 416, 416)
    y0, y1, y2 = m(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())

