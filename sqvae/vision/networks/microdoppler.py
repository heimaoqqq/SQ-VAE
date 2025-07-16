from networks.net_64 import EncoderVqResnet64, DecoderVqResnet64
from networks.net_256 import EncoderVqResnet256, DecoderVqResnet256


# 64×64版本 - 遵循原项目哲学
class EncoderVq_resnet(EncoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"


class DecoderVq_resnet(DecoderVqResnet64):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler"


# 256×256版本 - 高分辨率保持
class EncoderVq_resnet256(EncoderVqResnet256):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet256, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MicroDoppler"


class DecoderVq_resnet256(DecoderVqResnet256):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet256, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "MicroDoppler"
