from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/data/zhenglinyu/benchmarks/got10k/'
    settings.lasot_path = ''
    settings.network_path = '/home/zhenglinyu2/DCFST/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/data/zhenglinyu/benchmarks/nfs30/'
    settings.otb_path = '/data/zhenglinyu/benchmarks/otb100/'
    settings.results_path = '/home/zhenglinyu2/DCFST/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/zhenglinyu/benchmarks/trackingnet/'
    settings.uav_path = ''
    settings.vot18_path = '/data/zhenglinyu/benchmarks/vot18/'
    settings.vot19_path = '/data/zhenglinyu/benchmarks/vot19/'

    return settings

