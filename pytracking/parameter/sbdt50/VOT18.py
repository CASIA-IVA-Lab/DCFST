from pytracking.features import deep
from pytracking.utils import TrackerParams

def parameters(ID=None):

    # Tracker specific parameters
    params = TrackerParams()

    # Parameters for debugging
    params.output_image = False
    params.output_image_path = './debug/result_image/'

    # Parameters for device and tracking model
    params.use_gpu = True
    params.device = 'cuda'
    params.model = deep.SBDTNet50(net_path='DCFST-50.pth')

    # Parameters for sampling search region
    params.search_padding = 5.0        # Sampling size relative to target size
    params.img_sample_area = 288**2    # Area of the search region image

    # Parameters for training locator
    params.regularization = 0.1          # Regularization term to train locator (train with 0.1)
    params.learning_rate = 0.018         # Learning rate to update locator
    params.output_sigma_factor = 1/4     # Standard deviation of Gaussian label relative to target size (train with 1/4)
    params.proposals_num = 31**2         # Number of uniform proposals in locator (train with 31**2)
    params.train_skipping = 10           # How often to run locator training (common: 10)
    params.target_not_found = 0.0        # Absolute score threshold to detect target missing (small)
    params.init_samples_minimum_weight = 0.25    # Minimum weight of initial samples

    # Parameters for hard negative samples mining
    params.hard_negative_mining = True           # Whether to perform hard negative samples mining
    params.hard_negative_threshold = 0.5         # Relative threshold to find hard negative samples (common: 0.5)
    params.hard_negative_learning_rate = 0.22    # Learning rate if hard negative samples are detected (small)
    params.hard_negative_distance_ratio = 0.75   # Scope to ignore the detection of hard negative samples relative to target size

    # Parameters for window
    params.window_output = True         # Whether to perform window
    params.window_sigma_factor = 0.9    # Standard deviation of Gaussian window relative to target size (large)
    params.window_min_value = 0.3       # Min value of the output window (large)

    # Parameters for iounet refinement
    params.num_init_random_boxes = 19    # Number of random boxes for scale refinement (ATOM: 9)
    params.box_jitter_pos = 0.2          # How much to jitter the translation for random boxes (ATOM: 0.1)
    params.box_jitter_sz = 0.5           # How much to jitter the scale for random boxes (ATOM: 0.5)
    params.box_refinement_iter = 5       # Number of iterations for box refinement (ATOM: 5)
    params.maximal_aspect_ratio = 6      # Limit on the aspect ratio (ATOM: 6)
    params.iounet_k = 5                  # Top-k average to estimate final box (ATOM: 3)
    params.scale_damp = 0.6              # Linear interpolation coefficient for target scale update (small)

    # Parameters for data augmentation
    params.augmentation = True                   # Whether to perform data augmentation
    params.augmentation_expansion_factor = 2     # How much to expand sample when doing augmentation
    params.augmentation_method = {'fliplr': True, 
                                  'rotate': [5, -5, 10, -10, 20, -20], 
                                  'blur': [(2, 0.2), (0.2, 2), (3,1), (1, 3), (2, 2)]}

    return params