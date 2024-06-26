from py_utils.xywh2xyxy import xywh2xyxy
import numpy as np


def get_gt(file, dtype=int, convert=False, img: np = None):
	with open(file, 'r') as f:
		data = f.readlines()
		target_cls = []
		res = []
		for line in data:
			intlist = line.split()[1:]
			# intlist = list(map(type, line.split()[1:]))
			res.append(intlist)
			target_cls.append(np.array(line.split()[0], dtype=int))
		res = np.array(res, dtype=dtype)
	if convert:
		# for i, j in enumerate(res):
		# 	res[i] = xywh2xyxy(j)
		# img.shape => height,width
		res = xywh2xyxy(res)
		# print(img.shape())
		res[:, 0] *= img.shape[1]
		res[:, 1] *= img.shape[0]
		res[:, 2] *= img.shape[1]
		res[:, 3] *= img.shape[0]
	return target_cls, res.astype(int)


if __name__ == '__main__':
	# a = get_gt('test.txt')
	import cv2 as cv

	img_path = '/home/akuma/repos/dl/yolov7-akuma/datasets/Anti-UAV-jiafang/images/val/RGB.mp4_20240129_101734.526.jpg'
	anno_path = '/home/akuma/repos/dl/yolov7-akuma/datasets/Anti-UAV-jiafang/labels/val/RGB.mp4_20240129_101734.526.txt'
	img = cv.imread(img_path)
	a = get_gt(anno_path, float, True, img)
	print(a)
