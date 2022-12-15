class AOT_Args:
    def __init__(self) -> None:
        self.model = 'aotgan'
        self.block_num = 8
        self.rates = '1+2+4+8'
        self.gan_type = 'smgan'

        self.pre_train = './pre_trained/G0000000.pt'
        self.outputs = '../outputs'

class DeepFillv2_Args:
    def __init__(self) -> None:
        self.gan_type = 'WGAN'
        self.results_path = './results'
        self.gpu_ids = '0'
        self.cudnn_benchmark = True

        self.batch_size = 1
        self.in_channels = 4
        self.out_channels = 3
        self.latent_channels = 48
        self.pad_type = 'zero'
        self.activation = 'elu'
        self.norm = 'none'
        self.init_type = 'xavier'
        self.init_gain = 0.02
