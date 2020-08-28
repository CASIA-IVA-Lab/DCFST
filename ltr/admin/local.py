class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zhenglinyu2/SBDT/model/'     # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/data/zhenglinyu/LaSOT/source/LaSOTBenchmark/'
        self.got10k_dir = '/data/zhenglinyu/GOT/source/data/'
        self.trackingnet_dir = '/data/zhenglinyu/TrackingNet/source/'
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
