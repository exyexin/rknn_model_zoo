import numpy as np


def cal_iou(box1, box2):
	'''
	box: (N,4)
	e.g.[[1,2,3,4],
		[5,6,7,8],
		[...]]
	'''
	x11, y11, x12, y12 = np.split(box1, 4, axis=1)
	x21, y21, x22, y22 = np.split(box2, 4, axis=1)

	xa = np.maximum(x11, np.transpose(x21))
	xb = np.minimum(x12, np.transpose(x22))
	ya = np.maximum(y11, np.transpose(y21))
	yb = np.minimum(y12, np.transpose(y22))

	area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

	area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
	area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
	area_union = area_1 + np.transpose(area_2) - area_inter

	iou = area_inter / area_union
	return iou


if __name__ == '__main__':
	print(cal_iou(np.array([[500, 930, 656, 1069]]),
				  np.array([[501, 933, 650, 1062]])))
