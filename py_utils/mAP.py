import os
import cv2
import numpy as np


def txt_check(path):
	txt_type = ['txt']
	for _type in txt_type:
		if path.endswith(_type) or path.endswith(_type.upper()):
			return True
	return False


# def VOC2real(bbox):
# for line in source_file:  # 例遍 txt文件得每一行
# 	staff = line.split()  # 对每行内容 通过以空格为分隔符对字符串进行切片
# 	class_idx = int(staff[0])
#
# 	x_center, y_center, w, h = float(
# 		staff[1]) * width, float(staff[2]) * height, float(staff[3]) * width, float(staff[4]) * height
# 	x1 = round(x_center - w / 2)
# 	y1 = round(y_center - h / 2)
# 	x2 = round(x_center + w / 2)
# 	y2 = round(y_center + h / 2)
# return None

def error(msg):
	print(msg)


# sys.exit(0)

def get_true_anno(dir_path, img_size, co_helper):
	return None
	file_list = [os.path.join(dir_path, it) for it in sorted(os.listdir(dir_path))]
	label_list = []
	height, width = img_size
	bbox = []

	for path in file_list:
		if txt_check(path):
			label_list.append(path)

	for label_file in label_list:
		if os.path.basename(label_file)[0] != '2':
			continue

		with open(label_file, 'r') as f:  # 例遍 txt文件得每一行
			for line in f:
				staff = line.split()  # 对每行内容 通过以空格为分隔符对字符串进行切片
				class_idx = int(staff[0])

				x_center, y_center, w, h = float(
					staff[1]) * width, float(staff[2]) * height, float(staff[3]) * width, float(staff[4]) * height
				x1 = round(x_center - w / 2)
				y1 = round(y_center - h / 2)
				x2 = round(x_center + w / 2)
				y2 = round(y_center + h / 2)
				bbox.append([x1, y1, x2, y2])

	bbox = co_helper.get_real_box(bbox)

	return class_idx, bbox


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
	""" Compute the average precision, given the recall and precision curves.
	Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
	# Arguments
		tp:  True positives (nparray, nx1 or nx10).
		conf:  Objectness value from 0-1 (nparray).
		pred_cls:  Predicted object classes (nparray).
		target_cls:  True object classes (nparray).
		plot:  Plot precision-recall curve at mAP@0.5
		save_dir:  Plot save directory
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""

	# Sort by objectness
	i = np.argsort(-conf)
	tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

	# Find unique classes
	unique_classes = np.unique(target_cls)
	nc = unique_classes.shape[0]  # number of classes, number of detections

	# Create Precision-Recall curve and compute AP for each class
	px, py = np.linspace(0, 1, 1000), []  # for plotting
	ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
	for ci, c in enumerate(unique_classes):
		i = pred_cls == c
		n_l = (target_cls == c).sum()  # number of labels
		n_p = i.sum()  # number of predictions

		if n_p == 0 or n_l == 0:
			continue
		else:
			# Accumulate FPs and TPs
			fpc = (1 - tp[i]).cumsum(0)
			tpc = tp[i].cumsum(0)

			# Recall
			recall = tpc / (n_l + 1e-16)  # recall curve
			r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

			# Precision
			precision = tpc / (tpc + fpc)  # precision curve
			p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

			# AP from recall-precision curve
			for j in range(tp.shape[1]):
				ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
				if plot and j == 0:
					py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

	# Compute F1 (harmonic mean of precision and recall)
	f1 = 2 * p * r / (p + r + 1e-16)
	if plot:
		print("not be able to plot curve")
		if False:
			plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
			plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
			plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
			plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

	i = f1.mean(0).argmax()  # max F1 index
	return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
	""" Compute the average precision, given the recall and precision curves
	# Arguments
		recall:    The recall curve (list)
		precision: The precision curve (list)
		v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
	# Returns
		Average precision, precision curve, recall curve
	"""

	# Append sentinel values to beginning and end
	if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
		mrec = np.concatenate(([0.], recall, [1.0]))
	else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
		mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
	mpre = np.concatenate(([1.], precision, [0.]))

	# Compute the precision envelope
	mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

	# Integrate area under curve
	method = 'interp'  # methods: 'continuous', 'interp'
	if method == 'interp':
		x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
		ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
	else:  # 'continuous'
		i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
		ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

	return ap, mpre, mrec


def get_anno_img_path(gt_path, img_path, img_type='jpg'):
	gt_list = []
	img_list = []
	gt_lsdir = sorted(os.listdir(gt_path))
	# check file is txt
	for i in gt_lsdir:
		print(os.path.splitext(i)[-1])
		if os.path.splitext(i)[-1] == '.txt':
			gt_list.append(os.path.join(gt_path, i))

	for i in gt_list:
		# 获取图片名
		name = os.path.basename(i).replace('txt', img_type)
		# 获取图片路径+文件名
		name = os.path.join(img_path, name)
		if os.path.exists(name):
			img_list.append(name)

	return gt_list, img_list


def get_true_info(gt_list, img_list):
	# todo
	for i in gt_list:
		with open(i, 'r') as f:
			pass


def record_map(pred_boxes, pred_classes, pred_scores, gt_path, img_path, co_helper):
	# Todo 需要完成获取tp值的代码
	[gt_list, img_list] = get_anno_img_path(gt_path, img_path, 'jpg')

	target_cls, tp = get_true_anno(gt_path, img_path, co_helper=co_helper)
	output = ap_per_class(tp, pred_scores, pred_classes, target_cls)


if __name__ == '__main__':
	get_anno_img_path('../datasets/Anti-UAV-jiafang/labels', '../datasets/Anti-UAV-jiafang/val', 'jpg')
