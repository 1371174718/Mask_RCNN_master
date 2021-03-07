import os
import json

# { 'filename': '28503151_5b5b7ec140_b.jpg',
#   'regions': {
#       '0': {
#           'region_attributes': {},
#           'shape_attributes': {
#               'all_points_x': [...],
#               'all_points_y': [...],
#               'name': 'polygon'}},
#       ... more regions ...
#   },
#   'size': 100202
# }

def bigjson(RootDir,calcutype):
	jsondir = RootDir + '/{}_json'.format(calcutype)
	jsonresults = {}

	files = os.listdir(jsondir) # 所有json文件夹下的文件名
	for file in files: # 一个json代表一个图片，逐个图片读取
		if file.strip().split('.')[-1] == 'json': # 对json文件进行操作
			jsonresult = {'filename': '',
			              'regions': {}}
			filename = file.split('.')[0] + '.jpg'
			jsonresult['filename'] = filename
			jsonfile = json.load(open(os.path.join(jsondir,file))) # 打开
			print(jsonfile)
			shapes = jsonfile['shapes']# 包含region等信息
			numofshape = 0
			all_points_x = []
			all_points_y = []
			subjson = {'region_attributes': {},
			           'shape_attributes': {
				           'all_points_x': '',
				           'all_points_y': '',
				           'name': 'polygon'
			           }}
			for shape in shapes: # 每一个shape表示一个ROI区域
				if type(shape) is dict:
					points = shape['points'] # 一个ROI的所有点
					for point in points:
						all_points_x.append(point[0])
						all_points_y.append(point[1])
					subjson['shape_attributes']['all_points_x'] = all_points_x
					subjson['shape_attributes']['all_points_y'] = all_points_y
					jsonresult['regions'][numofshape] = subjson
					numofshape += 1
			jsonresults['{}'.format(filename)] = jsonresult
	with open(RootDir + '/{}/via_region_data.json'.format(calcutype), 'w') as f:
		# 设置不转换成ascii  json字符串首缩进
		f.write(json.dumps(jsonresults, ensure_ascii=False, indent=2))
	return jsonresults

if __name__ == '__main__':
	Rootdir = r'E:\GitHub_Projects\Mask_RCNN_master\samples\cylinderBlock\blockDataset'
	jsonresults = bigjson(Rootdir,'val')
	# 字典转换成json 存入本地文件
	print(jsonresults)
