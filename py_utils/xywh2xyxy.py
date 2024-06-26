def xywh2xyxy(box):
	box[:, 0] = box[:, 0] - box[:, 2] / 2.0
	box[:, 1] = box[:, 1] - box[:, 3] / 2.0
	box[:, 2] = box[:, 0] + box[:, 2]
	box[:, 3] = box[:, 1] + box[:, 3]
	return box
