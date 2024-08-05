import os
import cv2
import sys
import argparse
import numpy as np
import torch
from ultralytics.utils.metrics import box_iou
from py_utils.mAP import ap_per_class
from ultralytics.utils.ops import scale_coords,xywh2xyxy,xywhn2xyxy

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:realpath.index('rknn_model_zoo') + 1]))
from py_utils.coco_utils import COCO_test_helper

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

#tmp
names = {0:'UAV'}

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
		   "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow",
		   "elephant",
		   "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
		   "snowboard", "sports ball", "kite",
		   "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
		   "fork", "knife ",
		   "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut",
		   "cake", "chair", "sofa",
		   "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ",
		   "keyboard ", "cell phone", "microwave ",
		   "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ",
		   "hair drier", "toothbrush ")

# LABEL_SAVE_PATH = './datasets/Anti-UAV-jiafang/mAP_label'
ANNO_PATH = './datasets/Anti-UAV-jiafang/labels'


# def save_labels(boxes, classes, scores, path, labelname):
# 	# boxes = np.array2string(boxes, separator=',')
# 	s = zip(classes, scores, boxes)
# 	with open(os.path.join(path, labelname), 'w+') as f:
# 		for line in s:
# 			f.write(f'{CLASSES[line[0]]} {line[1]} {str(line[2])[1:-1]}\n')


def filter_boxes(boxes, box_confidences, box_class_probs):
	"""Filter boxes with object threshold.
	"""
	box_confidences = box_confidences.reshape(-1)
	candidate, class_num = box_class_probs.shape

	class_max_score = np.max(box_class_probs, axis=-1)
	classes = np.argmax(box_class_probs, axis=-1)

	if class_num == 1:
		_class_pos = np.where(box_confidences >= OBJ_THRESH)
		scores = (box_confidences)[_class_pos]
	else:
		_class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
		scores = (class_max_score * box_confidences)[_class_pos]

	boxes = boxes[_class_pos]
	classes = classes[_class_pos]

	return boxes, classes, scores


def nms_boxes(boxes, scores):
	"""Suppress non-maximal boxes.
	# Returns
		keep: ndarray, index of effective boxes.
	"""
	x = boxes[:, 0]
	y = boxes[:, 1]
	w = boxes[:, 2] - boxes[:, 0]
	h = boxes[:, 3] - boxes[:, 1]

	areas = w * h
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)

		xx1 = np.maximum(x[i], x[order[1:]])
		yy1 = np.maximum(y[i], y[order[1:]])
		xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
		yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

		w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
		h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
		inter = w1 * h1

		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= NMS_THRESH)[0]
		order = order[inds + 1]
	keep = np.array(keep)
	return keep


def box_process(position, anchors):
	grid_h, grid_w = position.shape[2:4]
	col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
	col = col.reshape(1, 1, grid_h, grid_w)
	row = row.reshape(1, 1, grid_h, grid_w)
	grid = np.concatenate((col, row), axis=1)
	stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

	col = col.repeat(len(anchors), axis=0)
	row = row.repeat(len(anchors), axis=0)
	anchors = np.array(anchors)
	anchors = anchors.reshape(*anchors.shape, 1, 1)

	box_xy = position[:, :2, :, :] * 2 - 0.5
	box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors

	box_xy += grid
	box_xy *= stride
	box = np.concatenate((box_xy, box_wh), axis=1)

	# Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
	xyxy = np.copy(box)
	xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
	xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
	xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
	xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y

	return xyxy


def post_process(input_data, anchors):
	boxes, scores, classes_conf = [], [], []
	# 1*255*h*w -> 3*85*h*w
	input_data = [_in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
	for i in range(len(input_data)):
		boxes.append(box_process(input_data[i][:, :4, :, :], anchors[i]))
		scores.append(input_data[i][:, 4:5, :, :])
		classes_conf.append(input_data[i][:, 5:, :, :])

	def sp_flatten(_in):
		ch = _in.shape[1]
		_in = _in.transpose(0, 2, 3, 1)
		return _in.reshape(-1, ch)

	boxes = [sp_flatten(_v) for _v in boxes]
	classes_conf = [sp_flatten(_v) for _v in classes_conf]
	scores = [sp_flatten(_v) for _v in scores]

	boxes = np.concatenate(boxes)
	classes_conf = np.concatenate(classes_conf)
	scores = np.concatenate(scores)

	# filter according to threshold
	boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

	# nms
	nboxes, nclasses, nscores = [], [], []

	for c in set(classes):
		inds = np.where(classes == c)
		b = boxes[inds]
		c = classes[inds]
		s = scores[inds]
		keep = nms_boxes(b, s)

		if len(keep) != 0:
			nboxes.append(b[keep])
			nclasses.append(c[keep])
			nscores.append(s[keep])

	if not nclasses and not nscores:
		return None, None, None

	boxes = np.concatenate(nboxes)
	classes = np.concatenate(nclasses)
	scores = np.concatenate(nscores)

	return boxes, classes, scores


def draw(image, boxes, scores, classes):
	for box, score, cl in zip(boxes, scores, classes):
		top, left, right, bottom = [int(_b) for _b in box]
		print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
		cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
		cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
					(top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
	model_path = args.model_path
	if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
		platform = 'pytorch'
		from py_utils.pytorch_executor import Torch_model_container
		model = Torch_model_container(args.model_path)
	elif model_path.endswith('.rknn'):
		platform = 'rknn'
		from py_utils.rknn_executor import RKNN_model_container
		model = RKNN_model_container(args.model_path, args.target, args.device_id)
	elif model_path.endswith('onnx'):
		platform = 'onnx'
		from py_utils.onnx_executor import ONNX_model_container
		model = ONNX_model_container(args.model_path)
	else:
		assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
	print('Model-{} is {} model, starting val'.format(model_path, platform))
	return model, platform


def img_check(path):
	img_type = ['.jpg', '.jpeg', '.png', '.bmp']
	for _type in img_type:
		if path.endswith(_type) or path.endswith(_type.upper()):
			return True
	return False


def process_label(label_path, label_name, img=None):
	# get label file
	label_file = os.path.join(label_path, label_name)
	lines=[]
	with open(label_file) as f:
		for line in f:
			data = list(map(float,line.strip().split()))
			lines.append(data)
	res = torch.tensor(lines)
	return res


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	# basic params
	parser.add_argument('--model_path', type=str, required=True, help='model path, could be .pt or .rknn file')
	parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
	parser.add_argument('--device_id', type=str, default=None, help='device id')

	parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
	parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

	# data params
	# coco val folder: '../../../datasets/COCO//val2017'
	parser.add_argument('--img_folder', type=str, default='./datasets/Anti-UAV-jiafang/val', help='img folder path')
	parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')
	parser.add_argument('--anchors', type=str, default='./RK_anchors.txt',
						help='target to anchor file, only yolov5, yolov7 need this param')
	parser.add_argument('--label_folder', type=str, default='')

	args = parser.parse_args()

	with open(args.anchors, 'r') as f:
		values = [float(_v) for _v in f.readlines()]
		anchors = np.array(values).reshape(3, -1, 2).tolist()
	print("use anchors from '{}', which is {}".format(args.anchors, anchors))

	# init model
	model, platform = setup_model(args)

	file_list = sorted(os.listdir(args.img_folder))
	img_list = []
	for path in file_list:
		if img_check(path):
			img_list.append(path)
	co_helper = COCO_test_helper(enable_letter_box=True)

	# param for ap_per_class
	tp = np.empty((0, 10))
	conf = np.empty(0)
	pred_cls = np.empty(0)
	target_cls = np.empty(0)

	# run test
	stats=[]
	for i in range(len(img_list)):
		print('infer {}/{}'.format(i + 1, len(img_list)), end='\r')

		img_name = img_list[i]
		img_path = os.path.join(args.img_folder, img_name)
		if not os.path.exists(img_path):
			print("{} is not found", img_name)
			continue

		img_src = cv2.imread(img_path)
		if img_src is None:
			continue

		# Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
		pad_color = (0, 0, 0)
		img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# preprocee if not rknn model
		if platform in ['pytorch', 'onnx']:
			input_data = img.transpose((2, 0, 1))
			input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
			input_data = input_data / 255.
		else:
			input_data = img

		outputs = model.run([input_data])
		boxes, classes, scores = post_process(outputs, anchors)
		
		pred = torch.cat((torch.from_numpy(boxes),torch.from_numpy(scores.reshape(-1,1)),torch.from_numpy(classes.reshape(-1,1))),dim=1)
		predn = pred.clone()
		scale_coords(img.shape[:2],predn[:,:2],img_src.shape[:2],None)
		scale_coords(img.shape[:2],predn[:,2:4],img_src.shape[:2],None)

		# cal mAP
		device = 'cpu'
		iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
		niou = iouv.numel()
		label_per_img = process_label(args.label_folder, img_name.replace('jpg','txt'), img_src)
		nl = len(label_per_img)
		#假定返回格式为numpy/torch
		tcls = label_per_img[:,0].tolist() if nl else []

		correct = torch.zeros(boxes.shape[0],niou,dtype=torch.bool,device=device)
		if nl:
			detected = []
			tcls_tensor = label_per_img[:,0]
			#是否进行xywh2xyxy?
			#需要将anno转化为图像中实际坐标
			tbox = label_per_img[:,1:5]
			tbox=xywhn2xyxy(tbox,w=img_src.shape[1],h=img_src.shape[0])
			for cls in torch.unique(tcls_tensor):
				ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
				pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
				if pi.shape[0]:
					ious,i=box_iou(predn[pi,:4],tbox[ti]).max(1)

				# Append detections
					detected_set = set()
					for j in (ious > iouv[0]).nonzero(as_tuple=False):
						d = ti[i[j]]  # detected target
						if d.item() not in detected_set:
							detected_set.add(d.item())
							detected.append(d)
							correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
							if len(detected) == nl:  # all targets already located in image
								break
		stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

	# Compute statistics
	stats = [np.concatenate(x, 0) for x in zip(*stats)]
	# p, r, ap, f1, ap_class = ap_per_class(tp, conf, pred_cls, target_cls)
	if len(stats) and stats[0].any():
		p, r, ap, f1, ap_class = ap_per_class(*stats,names=names)
		ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
		mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
		# nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
	else:
		# nt = torch.zeros(1)
		print("None, please checkout!")

	# Print results
	pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
	# print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
	print(pf % ('all', mp, mr, map50, map))

	# Print results per class
	# if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
	# 	for i, c in enumerate(ap_class):
	# 		print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
