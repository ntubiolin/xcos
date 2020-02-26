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
from model.model import xCosModel
insight_dir = "/home/r07944011/researches/InsightFace_Pytorch"
backbone_weights_path = 'work_space/save/model_2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
atten_weights_path = 'work_space/save/model_attention_2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
backbone_target_path = "work_space/save/model_ir_se50.pth"
output_name = '../pretrained_model/xcos/20200217_accu_9931_Arcface.pth'

backbone_weights_path = 'work_space/save/model_2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace.pth'
atten_weights_path = 'work_space/save/model_attention_2019-09-02-08-21_'\
                     'accuracy:0.9968333333333333_step:436692_CosFace.pth'
backbone_target_path = "work_space/save/model_irse50_CosFace_ms1m_9039.pth"
output_name = '../pretrained_model/xcos/20200226_accu_9968_Cosface.pth'
backbone_weights_path = op.join(insight_dir, backbone_weights_path)
atten_weights_path = op.join(insight_dir, atten_weights_path)
backbone_target_path = op.join(insight_dir, backbone_target_path)

xcos_model = xCosModel()
xcos_model.backbone.load_state_dict(torch.load(backbone_weights_path), strict=True)
xcos_model.attention.load_state_dict(torch.load(atten_weights_path), strict=True)
xcos_model.backbone_target.load_state_dict(torch.load(backbone_target_path))


model_state = xcos_model.state_dict()
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
