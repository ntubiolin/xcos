# from model.face_recog import Backbone_FC2Conv, Backbone
# from model.xcos_modules import XCosAttention
# backbone = Backbone_FC2Conv(50, 0.6, 'ir_se')
# attention = XCosAttention(use_softmax=True, softmax_t=1, chw2hwc=True)
# backbone_target = Backbone(50,
#                            0.6,
#                            'ir_se')


# backbone.load_state_dict(torch.load(backbone_weights_path), strict=True)
# attention.load_state_dict(torch.load(atten_weights_path), strict=True)
# backbone_target.load_state_dict(torch.load(backbone_target_path))

import os.path as op
import torch
from collections import OrderedDict
from model.model import NormalFaceModel
insight_dir = "/home/r07944011/researches/InsightFace_Pytorch"

backbone_path = "work_space/save/model_ir_se50.pth"
output_name = '../pretrained_model/baseline/20200228_accu_9952_Arcface_backbone.pth'

# backbone_path = "work_space/save/model_irse50_CosFace_ms1m_9039.pth"
# output_name = '../pretrained_model/baseline/20200228_accu_9930_Cosface_backbone.pth'
backbone_weights_path = op.join(insight_dir, backbone_path)

model = NormalFaceModel()
model.backbone.load_state_dict(torch.load(backbone_weights_path), strict=True)


model_state = model.state_dict()
model_state_tmp = OrderedDict()
for k, v in model_state.items():
    if k.startswith("head"):
        print(k)
    else:
        model_state_tmp[k] = v
model_state = model_state_tmp
state = {
    'state_dict': model_state
}
torch.save(state, output_name)
