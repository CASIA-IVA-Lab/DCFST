import os
import cv2
import sys
import pdb
import math
import time
import torch
import torch.nn
import torch.nn.functional as F

from pytracking import TensorList
from pytracking.features import augmentation
from pytracking.tracker.base import BaseTracker
from pytracking.utils.plotting import show_tensor
from pytracking.features.preprocessing import sample_patch
from pytracking.features.preprocessing import numpy_to_torch

class SBDT(BaseTracker):

	def initialize_model(self):
		'''
		if not getattr(self, 'model_initialized', False):
			self.params.model.initialize()
		self.model_initialized = True
		'''
		self.params.model.initialize() # for reproduce the VOT result

	# ------ MAIN INITIALIZE ------#
	def initialize(self, image, state, *args, **kwargs):
		self.frame_num = 1

		# For debug show only
		#image_show = image.copy() 

		# Fix random seed
		torch.manual_seed(1024)
		torch.cuda.manual_seed_all(1024)

		# Initialize features
		self.initialize_model()

		# Get position and size (y, x, h, w)
		self.target_pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
		self.target_sz = torch.Tensor([state[3], state[2]])
		self.initial_target_sz = self.target_sz.clone()

		# Set target scale and base target size (N)
		self.img_sample_sz = torch.Tensor([math.sqrt(self.params.img_sample_area)]) * torch.ones(2)
		self.target_sample_area = self.params.img_sample_area / self.params.search_area_scale**2

		# Get initial search area, sample scale ratio and target size in sample image 
		self.search_area = torch.prod(self.target_sz * self.params.search_area_scale)
		self.sample_scale = torch.sqrt(self.search_area / self.params.img_sample_area)
		self.target_sample_sz = self.target_sz / self.sample_scale

		# Generate centers of proposals for locator (N)
		self.proposals_xc, self.proposals_yc = self.init_proposal_centers_function()

		# Generate labels for locator (N)
		self.labels = self.init_labels_function()
		assert(self.labels.max().item()==1.0)

		# Creat output score window (N)
		self.output_window = None
		if getattr(self.params, 'window_output', True):
			self.output_window = self.init_output_window_function()

		# Setup scale bounds (N)
		self.min_scale_factor = self.params.min_scale_factor
		self.max_scale_factor = self.params.max_scale_factor

		# Extract initial transform samples
		im_tensor = numpy_to_torch(image)
		train_samples = self.generate_init_samples(im_tensor, self.target_pos, self.sample_scale).cuda()

		# Setup scale bounds (Martin)
		self.image_sz = torch.Tensor([im_tensor.shape[2], im_tensor.shape[3]])
		self.min_scale_factor = torch.max(10 / self.initial_target_sz)
		self.max_scale_factor = torch.min(self.image_sz / self.initial_target_sz)

		# Generate initial proposals for locator
		batch_size = train_samples.shape[0]
		init_proposals = self.get_locator_proposals(self.target_sample_sz)
		init_proposals = init_proposals.repeat(batch_size,1,1)

		# Feature Extract
		self.params.model.extract(train_samples, init_proposals)

		# Initialize iounet
		self.init_iou_net(self.target_pos, self.target_sz, self.sample_scale)

		# Initialize locator features
		self.initial_locator_features = self.params.model.locator_features.clone().mean(dim=0)
		self.locator_features_model = self.params.model.locator_features.clone().mean(dim=0)

		# Train locator model
		self.regularization_matrix = None
		self.locator_model = self.train_locator_model(self.locator_features_model)

		# Initial the hard negative sample detect region
		self.hard_negative_region_mask = self.init_hard_negative_region_function()

		# Initial the weight of first frame
		self.current_initial_frame_weight = 1.0

		# Output result image
		#self.output_result_image(image_show, state)

	# ------ MAIN TRACK ------#
	def track(self, image):
		self.frame_num += 1

		# For debug show only
		#image_show = image.copy()
		
		# Conver to tensor and GPU
		image_cuda = self.numpy_to_tensor_gpu(image)

		# ------- LOCALIZATION ------- #
		sample_pos = self.target_pos.clone()
		sample_scale = self.sample_scale.clone()
		target_sample_sz = self.target_sample_sz.clone()

		# sample and extract features
		test_sample = sample_patch(image_cuda, sample_pos, sample_scale*self.img_sample_sz, self.img_sample_sz)
		test_locator_proposals = self.get_locator_proposals(target_sample_sz)
		self.params.model.extract(test_sample, test_locator_proposals)

		# calcualte the localization score
		test_locator_score = torch.mm(self.params.model.locator_features, self.locator_model)
		if getattr(self.params, 'window_output', False):
			test_locator_score = test_locator_score * self.output_window
		max_score, max_id = torch.max(test_locator_score, dim=0)
		max_score, max_id = max_score.item(), max_id.item()

		# when target not found
		if max_score < self.params.target_not_found_threshold:
			# maintain the original target position and size
			new_state = torch.cat((self.target_pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))
			# Output result image
			#self.output_result_image(image_show, new_state)
			return new_state.tolist()

		# update the target position
		self.target_pos[0] = self.target_pos[0] + (self.proposals_yc[max_id].item() - self.img_sample_sz[1]*0.5) * sample_scale
		self.target_pos[1] = self.target_pos[1] + (self.proposals_xc[max_id].item() - self.img_sample_sz[0]*0.5) * sample_scale

		# refine the target position and size by IoUNet
		new_pos, new_target_sz = self.refine_target_box(self.target_pos, self.target_sz, sample_pos, sample_scale)

		# bound the taeget size
		if new_target_sz is not None:
		    new_target_sz = torch.min(new_target_sz, self.initial_target_sz*self.max_scale_factor)
		    new_target_sz = torch.max(new_target_sz, self.initial_target_sz*self.min_scale_factor)

		# update the target and sampling message
		if new_pos is not None:
			self.target_pos = new_pos.clone()
			self.target_sz = new_target_sz.clone()
			self.search_area = torch.prod(self.target_sz * self.params.search_area_scale)
			self.sample_scale = torch.sqrt(self.search_area / self.params.img_sample_area)
			self.target_sample_sz = self.target_sz / self.sample_scale

		# Return new state
		new_state = torch.cat((self.target_pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

		# Output result image
		#self.output_result_image(image_show, new_state)

		# ------- UPDAT MODEL------- #
		train_sample = sample_patch(image_cuda, self.target_pos, self.sample_scale*self.img_sample_sz, self.img_sample_sz)
		train_locator_proposals = self.get_locator_proposals(self.target_sample_sz)
		self.params.model.extract(train_sample, train_locator_proposals, only_locator=True)

		hard_flag = False
		if self.params.hard_negative_mining:
			train_locator_score = torch.mm(self.params.model.locator_features, self.locator_model)
			train_locator_score = train_locator_score * self.hard_negative_region_mask
			max_score, _ = torch.max(train_locator_score, dim=0)
			if max_score > self.params.hard_negative_threshold:
				hard_flag = True

		if hard_flag:
			learning_rate = self.params.hard_negative_learning_rate
		else:
			learning_rate = self.params.learning_rate

		self.locator_features_model = (1 - learning_rate) * self.locator_features_model + learning_rate * self.params.model.locator_features
		self.current_initial_frame_weight = (1 - learning_rate) * self.current_initial_frame_weight

		if self.current_initial_frame_weight < self.params.init_samples_minimum_weight:
			diff = self.params.init_samples_minimum_weight - self.current_initial_frame_weight
			coff = diff / (1 - self.current_initial_frame_weight)
			self.locator_features_model = (1 - coff) * self.locator_features_model + coff * self.initial_locator_features
			self.current_initial_frame_weight = self.params.init_samples_minimum_weight

		if (self.frame_num % self.params.train_skipping == 0) or (hard_flag):
			self.locator_model = self.train_locator_model(self.locator_features_model, self.locator_model)

		return new_state.tolist()

	def numpy_to_tensor_gpu(self, image):
		image = torch.from_numpy(image)
		image = image.cuda()
		image = image.permute(2,0,1).unsqueeze(0).to(torch.float32)
		return image

	def init_proposal_centers_function(self):
		search_area_scale = self.params.search_area_scale
		num_proposals = self.params.num_proposals_locator
		num_proposals_sqrt = int(math.sqrt(num_proposals))

		WIDTH, HEIGHT = self.img_sample_sz[0], self.img_sample_sz[1]
		x_step = ((WIDTH - WIDTH/search_area_scale) / (num_proposals_sqrt-1))
		y_step = ((HEIGHT - HEIGHT/search_area_scale) / (num_proposals_sqrt-1))

		proposals_xc = torch.arange(num_proposals_sqrt).repeat(num_proposals_sqrt).type(torch.float32)
		proposals_yc = torch.arange(num_proposals_sqrt).repeat(num_proposals_sqrt,1).t().reshape(-1).type(torch.float32)

		proposals_xc = proposals_xc * x_step + WIDTH/(search_area_scale*2)
		proposals_yc = proposals_yc * y_step + HEIGHT/(search_area_scale*2)

		proposals_xc = proposals_xc.to(self.params.device)
		proposals_yc = proposals_yc.to(self.params.device)
		return proposals_xc, proposals_yc

	def init_labels_function(self):
		proposals_xc = self.proposals_xc
		proposals_yc = self.proposals_yc
		sigma_factor = self.params.output_sigma_factor
		target_sample_area = self.target_sample_area

		WIDTH, HEIGHT = self.img_sample_sz[0], self.img_sample_sz[1]
		x_dist = proposals_xc - (WIDTH * 0.5).item()
		y_dist = proposals_yc - (HEIGHT * 0.5).item()

		sigma = sigma_factor * math.sqrt(target_sample_area)
		labels = torch.exp(-0.5 * (x_dist.pow(2)+y_dist.pow(2))/sigma**2)
		labels = labels.to(self.params.device).reshape(-1,1)
		return labels

	def init_output_window_function(self):
		proposals_xc = self.proposals_xc
		proposals_yc = self.proposals_yc
		window_min = self.params.window_min
		sigma_factor = self.params.window_sigma_factor
		target_sample_area = self.target_sample_area

		WIDTH, HEIGHT = self.img_sample_sz[0], self.img_sample_sz[1]
		x_dist = proposals_xc - (WIDTH * 0.5).item()
		y_dist = proposals_yc - (HEIGHT * 0.5).item()

		sigma = sigma_factor * math.sqrt(target_sample_area)
		output_window = torch.exp(-0.5 * (x_dist.pow(2)+y_dist.pow(2))/sigma**2)
		output_window = output_window.clamp(window_min)
		output_window = output_window.to(self.params.device).reshape(-1,1)
		return output_window

	def init_hard_negative_region_function(self):
		proposals_xc = self.proposals_xc
		proposals_yc = self.proposals_yc
		img_sample_area = self.params.img_sample_area
		distance_ratio = self.params.hard_negative_distance_ratio

		region_mask = torch.zeros(proposals_xc.shape, device=self.params.device)
		x_distance = proposals_xc - (self.img_sample_sz[0] * 0.5).item()
		y_distance = proposals_yc - (self.img_sample_sz[1] * 0.5).item()
		distance = torch.sqrt(x_distance.pow(2)+y_distance.pow(2))
		distance_threshold = math.sqrt(img_sample_area * distance_ratio**2)
		region_mask[distance>distance_threshold] = 1.0
		region_mask = region_mask.view(-1,1)
		return region_mask

	def generate_init_samples(self, im: torch.Tensor, target_pos, sample_scale) -> TensorList:
		"""Generate augmented initial samples."""

		# Compute augmentation size
		aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
		aug_expansion_sz = self.img_sample_sz.clone()
		aug_output_sz = None
		if aug_expansion_factor is not None and aug_expansion_factor != 1:
			aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
			aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
			aug_expansion_sz = aug_expansion_sz.float()
			aug_output_sz = self.img_sample_sz.long().tolist()

		# Random shift operator
		get_rand_shift = lambda: None

		# Create transofmations
		self.transforms = [augmentation.Identity(aug_output_sz)]
		if 'shift' in self.params.augmentation:
			self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation['shift']])
		if 'relativeshift' in self.params.augmentation:
			get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
			self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation['relativeshift']])
		if 'fliplr' in self.params.augmentation and self.params.augmentation['fliplr']:
			self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
		if 'blur' in self.params.augmentation:
			self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation['blur']])
		if 'scale' in self.params.augmentation:
			self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation['scale']])
		if 'rotate' in self.params.augmentation:
			self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation['rotate']])

		init_sample = sample_patch(im, target_pos, sample_scale*aug_expansion_sz, aug_expansion_sz)
		init_samples = torch.cat([T(init_sample) for T in self.transforms])
		if not self.params.use_augmentation:
			init_samples = init_samples[0:1,...]
		return init_samples

	def init_iou_net(self, target_pos, target_sz, sample_scale):
		# Setup IoU net
		self.iou_predictor = self.params.model.iou_predictor
		for p in self.iou_predictor.parameters():
			p.requires_grad = False

		# Get target boxes and convert
		target_boxes = self.get_iounet_box(target_pos, target_sz, target_pos.round(), sample_scale)
		target_boxes = target_boxes.unsqueeze(0).to(self.params.device)

		# Get iou backbone features
		iou_backbone_features = self.params.model.iounet_backbone_features

		# Remove other augmentations such as rotation
		iou_backbone_features = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_features])

		# Extract target IoU feat
		with torch.no_grad():
			target_iou_feat = self.iou_predictor.get_filter(iou_backbone_features, target_boxes)
		self.target_iou_feat = TensorList([x.detach().mean(0) for x in target_iou_feat])

	def get_iounet_box(self, target_pos, target_sz, sample_pos, sample_scale):
		"""All inputs in original image coordinates"""
		box_center = (target_pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
		box_sz = target_sz / sample_scale
		target_ul = box_center - (box_sz - 1) / 2
		return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

	def get_locator_proposals(self, target_sample_sz):
		proposals_xc = self.proposals_xc
		proposals_yc = self.proposals_yc
		num_proposals_locator = self.params.num_proposals_locator

		proposals = torch.zeros(num_proposals_locator, 4, device=self.params.device)
		proposals[:,0] = proposals_xc - (target_sample_sz[1]*0.5).item()
		proposals[:,1] = proposals_yc - (target_sample_sz[0]*0.5).item()
		proposals[:,2] = target_sample_sz[1].item()
		proposals[:,3] = target_sample_sz[0].item()
		return proposals.unsqueeze(0)

	def train_locator_model(self, locator_features, model=None):
		regularization = self.params.regularization
		if self.regularization_matrix is None:
			self.regularization_matrix = regularization*torch.eye(locator_features.shape[1], device=self.params.device)

		train_XTX = torch.mm(locator_features.t(), locator_features)
		train_XTX = train_XTX + self.regularization_matrix
		train_XTY = torch.mm(locator_features.t(), self.labels)

		if model is None:
			model = torch.potrs(train_XTY, torch.potrf(train_XTX))
		else:
			for _ in range(30):
				model, _ = torch.trtrs(train_XTY - torch.mm(torch.triu(train_XTX, diagonal=1), model), torch.tril(train_XTX, diagonal=0), upper=False)
		return model

	def refine_target_box(self, target_pos, target_sz, sample_pos, sample_scale):
		# Initial box for refinement
		init_box = self.get_iounet_box(target_pos, target_sz, sample_pos, sample_scale)

		# Extract features from the relevant scale
		iou_features = self.params.model.iounet_features
		iou_features = TensorList([x[0:1,...] for x in iou_features])

		init_boxes = init_box.view(1,4).clone()
		if self.params.num_init_random_boxes > 0:
			# Get random initial boxes
			square_box_sz = init_box[2:].prod().sqrt()
			rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])
			minimal_edge_size = init_box[2:].min()/3
			rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
			new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
			new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
			init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
			init_boxes = torch.cat([init_box.view(1,4), init_boxes])

		# Refine boxes by maximizing iou
		output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

		# Remove weird boxes with extreme aspect ratios
		output_boxes[:, 2:].clamp_(1)
		aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
		keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
		output_boxes = output_boxes[keep_ind,:]
		output_iou = output_iou[keep_ind]

		# If no box found
		if output_boxes.shape[0] == 0:
			return None, None

		# Take average of top k boxes
		k = getattr(self.params, 'iounet_k', 5)
		topk = min(k, output_boxes.shape[0])
		_, inds = torch.topk(output_iou, topk)
		predicted_box = output_boxes[inds, :].mean(0).cpu()
		predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0).cpu()

		# Update position
		new_pos = predicted_box[:2] + predicted_box[2:]/2 - (self.img_sample_sz - 1) / 2
		new_pos = new_pos.flip((0,)) * sample_scale + sample_pos

		# Linear interpolation to update the target size
		new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
		new_target_sz = self.params.scale_damp * self.target_sz + (1 - self.params.scale_damp) * new_target_sz

		return new_pos, new_target_sz

	def optimize_boxes(self, iou_features, init_boxes):
		# Optimize iounet boxes
		output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
		step_length = self.params.box_refinement_step_length

		for i_ in range(self.params.box_refinement_iter):
			# forward pass
			bb_init = output_boxes.clone().detach()
			bb_init.requires_grad = True

			outputs = self.iou_predictor.predict_iou(self.target_iou_feat, iou_features, bb_init)

			if isinstance(outputs, (list, tuple)):
				outputs = outputs[0]

			outputs.backward(gradient = torch.ones_like(outputs))

			# Update proposal
			output_boxes = bb_init + step_length * (bb_init.grad*100).round()/100 * bb_init[:, :, 2:].repeat(1, 1, 2)
			output_boxes.detach_()

			step_length *= self.params.box_refinement_step_decay

		return output_boxes.view(-1,4), outputs.detach().view(-1)

	def output_result_image(self, image, state):
		if self.params.output_image:
			if not os.path.exists(self.params.output_image_path):
				os.mkdir(output_dir)
			cv2.rectangle(image, (int(state[0]),int(state[1])),(int(state[0]+state[2]),int(state[1]+state[3])), (255,0,0), 3)
			cv2.imwrite(os.path.join(output_dir,'{}.jpg'.format(self.frame_num)), cv.cvtColor(image, cv.COLOR_RGB2BGR))

'''
import cv2
for i in range(train_samples.shape[0]):
	output_dir = '/home/zhenglinyu2/SBDT_tracking/debug/transform_image/'
	count = len(os.listdir(output_dir))
	transform_image = train_samples[i,...].permute(1,2,0)
	transform_image = transform_image.data.numpy()
	cv2.imwrite(os.path.join(output_dir,'{}.jpg'.format(count+1)),transform_image))
'''

'''
torch.cuda.synchronize()
start = time.time()
torch.cuda.synchronize()
print(time.time() - start)
'''
