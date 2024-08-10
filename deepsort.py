import argparse
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_coords,xywhn2xyxy, scale_image
from ultralytics.utils.plotting import ImageDraw
import cv2
import numpy as np

from rknnlite.api import RKNNLite as RKNN
# from rknnlite.api import RKNN

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE=(640,640)

CLASSES = ("UAV")

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/rknn-onnx/Anti-UAV-jiafang4-fp.rknn', help='model path, could be .pt or .rknn file')
	parser.add_argument('--target', type=str, default='onboard', help='target RKNPU platform')
	parser.add_argument('--device_id', type=str, default=None, help='device id')

	# parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
	# parser.add_argument('--img_save', action='store_true', default=False, help='save the result')
	parser.add_argument('--video_save', action='store_true', default=False, help='save the result')
	parser.add_argument('--video_show', action='store_true', default=False, help='show the result')

	# data params
	parser.add_argument('--video', type=str, default='./visible.mp4', help='video path')
	parser.add_argument('--anchors', type=str, default='./RK_anchors.txt',
						help='target to anchor file, only yolov5, yolov7 need this param')
	
	return parser.parse_args()

class RKNN_model_container():
	def __init__(self, model_path, target=None, device_id=None) -> None:
		rknn = RKNN(verbose=True)

		# Direct Load RKNN Model
		rknn.load_rknn(model_path)

		print('--> Init runtime environment')
		if target==None:
			ret = rknn.init_runtime()
		elif target=='onboard':
			ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0)
		else:
			ret = rknn.init_runtime(target=target, device_id=device_id)
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		print('done')
		
		self.rknn = rknn 

	def run(self, inputs):
		if isinstance(inputs, list) or isinstance(inputs, tuple):
			pass
		else:
			inputs = [inputs]

		result = self.rknn.inference(inputs=inputs)
	
		return result
	
class Video():
	def __init__(self,path:str) -> None:
		self.video=path
		self.cap = cv2.VideoCapture(self.video)
	
	def __iter__(self):
		return self

	def __next__(self):
		ret, self.img0 = self.cap.read()
		if not ret:
			print("read video frame error!")
		else:
			letterbox = LetterBox(IMG_SIZE)
			self.img1 = letterbox(image=self.img0)
			cv2.cvtColor(self.img1,cv2.COLOR_BGR2RGB)
		return self.img0, self.img1

def setup_model(args):
	model_path = args.model_path
	if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
		platform = 'pytorch'
		from py_utils.pytorch_executor import Torch_model_container
		model = Torch_model_container(args.model_path)
	elif model_path.endswith('.rknn'):
		platform = 'rknn'
		# from py_utils.rknn_executor import RKNN_model_container
		model = RKNN_model_container(args.model_path, args.target, args.device_id)
	elif model_path.endswith('onnx'):
		platform = 'onnx'
		from py_utils.onnx_executor import ONNX_model_container
		model = ONNX_model_container(args.model_path)
	else:
		assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
	print('Model-{} is {} model, starting val'.format(model_path, platform))
	return model, platform

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
	# todo
	for box, score, cl in zip(boxes, scores, classes):
		top, left, right, bottom = [int(_b) for _b in box]
		print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
		cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
		cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
					(top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if __name__ == '__main__':
	args = get_args()
	with open(args.anchors, 'r') as f:
		values = [float(_v) for _v in f.readlines()]
		anchors = np.array(values).reshape(3, -1, 2).tolist()
	print("use anchors from '{}', which is {}".format(args.anchors, anchors))
	model, platform = setup_model(args)
	video = Video(args.video)
	for img_src, img1 in video:
		# preprocee if not rknn model
		if platform in ['pytorch', 'onnx']:
			input_data = img1.transpose((2, 0, 1))
			input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
			input_data = input_data / 255.
		else:
			input_data = np.expand_dims(img1,axis=0)

		# print(input_data.shape)
		outputs =model.run([input_data])

		boxes, classes, scores = post_process(outputs, anchors)
		print(type(boxes),len(boxes))
		pred = torch.cat((torch.from_numpy(boxes),torch.from_numpy(scores.reshape(-1,1)),torch.from_numpy(classes.reshape(-1,1))),dim=1)
		# 处理预测后的标签 -> ltbr(img_src)
		predn = pred.clone()
		scale_coords(img1.shape[:2],predn[:,:2],img_src.shape[:2],None)
		scale_coords(img1.shape[:2],predn[:,2:4],img_src.shape[:2],None)
		# draw bbox
		img_draw = img_src.copy()
		draw(img_draw,predn[:,:4],predn[:,4],classes)
		print("draw Done!")
		cv2.imshow("video.jpg",img_draw)
		cv2.waitKey(1)
		
