import os
import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from thop import profile
from thop import clever_format

torch.backends.cudnn.benchmark = True

from config import GlobalConfig
from model import VitFuser
from data import CARLA_Data
from utils import load_weight
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.empty_cache()

#  ===  hyperparameters  === 
BATCH_SIZE = 64
PTH_PATH = '/home/gyp/program/my_transfuser/transfuser/vitfuser/log/vitfuser_pvt/best_model.pth'
DEV = "cuda"
REPEAT = 1000
#  ========================= 

class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10

	def validate(self):
		model.eval()

		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.

			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				
				# create batch and move to GPU
				fronts_in = data['fronts']
				lefts_in = data['lefts']
				rights_in = data['rights']
				rears_in = data['rears']
				lidars_in = data['lidars']
				fronts = []
				lefts = []
				rights = []
				rears = []
				lidars = []
				for i in range(config.seq_len):
					fronts.append(fronts_in[i].to(DEV, dtype=torch.float32))
					if not config.ignore_sides:
						lefts.append(lefts_in[i].to(DEV, dtype=torch.float32))
						rights.append(rights_in[i].to(DEV, dtype=torch.float32))
					if not config.ignore_rear:
						rears.append(rears_in[i].to(DEV, dtype=torch.float32))
					lidars.append(lidars_in[i].to(DEV, dtype=torch.float32))

				# driving labels
				command = data['command'].to(DEV)
				gt_velocity = data['velocity'].to(DEV, dtype=torch.float32)
				gt_steer = data['steer'].to(DEV, dtype=torch.float32)
				gt_throttle = data['throttle'].to(DEV, dtype=torch.float32)
				gt_brake = data['brake'].to(DEV, dtype=torch.float32)

				# target point
				target_point = torch.stack(data['target_point'], dim=1).to(DEV, dtype=torch.float32)
				pred_wp = model(fronts+lefts+rights+rears, lidars, target_point, gt_velocity)

				gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(DEV, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
				gt_waypoints = torch.stack(gt_waypoints, dim=1).to(DEV, dtype=torch.float32)
				wp_epoch += float(F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean())

				num_batches += 1
					
			wp_loss = wp_epoch / float(num_batches)
			print(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.5f}')
			
			self.val_loss.append(wp_loss)


# Config
config = GlobalConfig()

# Dataset
val_set = CARLA_Data(root=config.val_data, config=config)
dataloader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = VitFuser(config, DEV)
print(model)
load_weight(model, PTH_PATH)
# model.eval()
# Engine
engine = Engine()
engine.validate()

# profile
image_input = torch.randn(1, 3, 256, 256).to(DEV)
lidar_input = torch.randn(1, 2, 256, 256).to(DEV)
target_point = torch.randn(1, 2).to(DEV)
gt_velocity = torch.randn(1).to(DEV)
flops, params = profile(model, inputs=([image_input], [lidar_input], target_point, gt_velocity))
flops, params = clever_format([flops, params], "%.3f")
print(f'flops: {flops}, params: {params}')

# warm up
for _ in range(100):
	pred_wp = model([image_input], [lidar_input], target_point, gt_velocity)
# speed test
time_start = time.time()
for _ in range(REPEAT):
	pred_wp = model([image_input], [lidar_input], target_point, gt_velocity)
	torch.cuda.synchronize()
time_end = time.time()
print('latency: ', (time_end-time_start)/REPEAT * 1000, 'ms')

# with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=True) as prof:
#     model([image_input], [lidar_input], target_point, gt_velocity)
# resultList = prof.table().split("\n")
# prof.export_chrome_trace('track.json')
# prof.export_stacks('cpu_stack.json', metric="self_cpu_time_total")
# prof.export_stacks('gpu_stack.json', metric="self_cuda_time_total")