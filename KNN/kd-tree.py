#encoding=utf-8


class KdNode(object):
	def __init__(self, dom_elt, split, left, right):
		self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
		self.split = split  # 整数（进行分割维度的序号）
		self.left = left  # 该结点分割超平面左子空间构成的kd-tree
		self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree(object):
	def __init__(self, data):
		self.k = len(data[0])
		self.root = self.create_node(0, data)

	def create_node(self, split, data_set):
		pass

