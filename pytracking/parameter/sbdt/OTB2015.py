from pytracking.features import deep
from pytracking.utils import TrackerParams

def parameters(ID=None):

    # Tracker specific parameters
    params = TrackerParams()

    # ------------------ CHANGED ------------------#
    
    # Output result images
    params.output_image = False
    params.output_image_path = './debug/result_image/'

    # Training parameters for locator
    params.regularization = 0.1                  # Regularization term to train locator
    params.learning_rate = 0.013                 # Learning rate to update locator features model
    params.train_skipping = 10                   # How often to run training (every n-th frame)
    params.output_sigma_factor = 1/4             # Standard deviation of Gaussian label relative to target size
    params.target_not_found_threshold = 0.25     # Absolute score threshold to detect target missing
    params.init_samples_minimum_weight = 0.25    # Minimum weight of initial samples

    # Hard negative samples mining
    params.hard_negative_mining = True            # Perform hard negative samples mining
    params.hard_negative_threshold = 0.3          # Absolute threshold to find hard negative samples
    params.hard_negative_learning_rate = 0.125    # Learning rate if hard negative samples are detected
    params.hard_negative_distance_ratio = 0.15    # Detect hard negative samples range relative to image sample area

    # Windowing
    params.window_output = True         # Perform windowing to output scores
    params.window_sigma_factor = 1.05   # Standard deviation of Gaussian output window relative to target size
    params.window_min = 0.5             # Min value of the output window

    # Scale update
    params.scale_damp = 0.7    # Linear interpolation coefficient for target scale update

    # Setup the tracking model
    params.model = deep.SBDTNet18(net_path='DCFST-18.pth')

    # GPU
    params.use_gpu = True
    params.device = 'cuda'

    # Patch sampling
    params.search_area_scale = 5      # Scale relative to target size
    params.img_sample_area = 288**2   # Area of the image sample

    # Locator proposals
    params.num_proposals_locator = 31**2    # Number of proposals in locator

    # Data augmentation
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20],
                           'blur': [(2, 0.2), (0.2, 2), (3,1), (1, 3), (2, 2)]}
    params.augmentation_expansion_factor = 2    # How much to expand sample when doing augmentation
    params.use_augmentation = True              # Whether to use augmentation

    # IoUNet
    params.iounet_k = 3                      # Top-k average to estimate final box
    params.num_init_random_boxes = 9         # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1              # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5               # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6          # Limit on the aspect ratio
    params.box_refinement_iter = 5           # Number of iterations for refining the boxes
    params.box_refinement_step_length = 1    # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1     # Multiplicative step length decay (1 means no decay)

    # Scale bounds
    params.min_scale_factor = 0.2    # Min value of the scale bound
    params.max_scale_factor = 5.0    # Max value of the scale bound

    return params