import torch
from torch import optim
import numpy as np
import sklearn.metrics as metrics




def Max_Index(array):
	max_index = 0
	for i in range(len(array)):
		if(array[i]>array[max_index]):
			max_index = i
	return max_index

def Get_Report(true_labels, pred_labels, labels=None, digits=4):
	recall = metrics.recall_score(true_labels, pred_labels, average='macro')
	precision = metrics.precision_score(true_labels, pred_labels, average='macro')
	macrof1 = metrics.f1_score(true_labels, pred_labels, average='macro')
	microf1 = metrics.f1_score(true_labels, pred_labels, average='micro')
	acc = metrics.accuracy_score(true_labels, pred_labels)
	return recall, precision, macrof1, microf1, acc


class C2F(torch.nn.Module):
	def __init__(self, len1, len2, len3):
		super(C2F, self).__init__()

		self.height = 10
		self.width  = 100

		# self.fc = torch.nn.Linear( in_features=999, out_features=99 )

		self.rnn = torch.nn.RNN(input_size=100, hidden_size=50, num_layers=1, batch_first=True, bidirectional=True)

		self.convs = torch.nn.ModuleList([
			torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(h, 100), stride=(1, self.width), padding=0),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=(self.height - h + 1, 1), stride=(self.height - h + 1, 1))
				) for h in [2, 3, 4]])

		self.gate1 = torch.autograd.Variable(torch.randn(1, 12))
		self.gate2 = torch.autograd.Variable(torch.randn(1, 50))
		self.gate3 = torch.autograd.Variable(torch.randn(1, 50))

		self.ST1_fc1 = torch.nn.Linear(212, 50)
		self.ST1_fc2 = torch.nn.Linear(50, len1)

		self.ST2_fc1 = torch.nn.Linear(212 + 50, 50)
		self.ST2_fc2 = torch.nn.Linear(50, len2)

		self.ST3_fc1 = torch.nn.Linear(212 + 100 + 50 + 1, 50)
		self.ST3_fc2 = torch.nn.Linear(50, len3)

	def CNNRNN_Encoder(self, x):
		xd = torch.cat([conv(x) for conv in self.convs], dim=1)
		xd = xd.view(-1, xd.size(1))
		xd = xd * self.gate1
		# print(xd.shape)

		xu = x.view(-1, 10, 100)
		xu, _ = self.rnn(xu, None)
		# print(xu.shape)

		xu_left = xu[:, 0, :]
		xu_right = xu[:, -1, :]
		# print(xu_left.shape)
		# print(xu_right.shape)

		xu = torch.cat([xu_left, xu_right], 1)
		# print(xu.shape)

		out = torch.cat([xu, xd], 1)

		return out

	def coarse_forward1(self, x1):
		# (32, 212)
		x1e = self.CNNRNN_Encoder(x1)
		# print(x1e.shape)

		x1e_fc1 = torch.nn.functional.relu(self.ST1_fc1(x1e))
		# print(x1e_fc1.shape, ".")
		x1e_fc2 = self.ST1_fc2(x1e_fc1)
		# print(x1e_fc2.shape)

		return x1e_fc2

	def coarse_forward2(self, x2):
		# (32, 212)
		x2e = self.CNNRNN_Encoder(x2)
		# print(x2e.shape, ".")

		x2e_fc1 = torch.nn.functional.relu(self.ST1_fc1(x2e))
		# print(x2e_fc1.shape, ".")
		x2e_fc1 = x2e_fc1 * self.gate2
		# print(x2e_fc1.shape, ".")

		x2ec = torch.cat([x2e, x2e_fc1], 1)
		# print(x2ec.shape)

		x2ec_fc1 = torch.nn.functional.relu(self.ST2_fc1(x2ec))
		x2ec_fc2 = self.ST2_fc2(x2ec_fc1)
		# print(x2ec_fc2.shape)

		return x2ec_fc2

	def fine_forward3(self, x3s, x3w):
		# (32, 212)
		x3s = self.CNNRNN_Encoder(x3s)
		# print(x3s.shape, ".")

		x3se_fc1 = torch.nn.functional.relu(self.ST1_fc1(x3s))
		# print(x3se_fc1.shape, ".")

		x3se_fc1 = x3se_fc1 * self.gate3
		# print(x3se_fc1.shape, ".")

		x3ec = torch.cat([x3s, x3w], 1)
		x3ec = torch.cat([x3ec, x3se_fc1], 1)
		# print(x3ec.shape, ".")

		x3ec_fc1 = torch.nn.functional.relu(self.ST3_fc1(x3ec))
		# print(x3ec_fc1.shape, ".")

		x3ec_fc2 = self.ST3_fc2(x3ec_fc1)
		# print(x3ec_fc2.shape, ".")

		return x3ec_fc2

	def train(self, train_x1, train_y1, test_x1,  test_y1, train_x2, train_y2, test_x2,  test_y2, train_x3s, train_x3w, train_y3, test_x3s,  test_x3w,  test_y3):

		criterion = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
		optimizer = optim.Adam(self.parameters(), lr=0.00001)

		v_test_x1  = torch.autograd.Variable(torch.Tensor(np.array([[obj] for obj in test_x1])))

		for epoch in range(100):
			optimizer.zero_grad()

			rand_index_x1 = np.random.choice(len(train_x1), size=32, replace=False)

			batch_x1 = torch.autograd.Variable(torch.Tensor(np.array([[obj] for i, obj in enumerate(train_x1)  if i in rand_index_x1])))
			batch_y1 = torch.autograd.Variable(torch.LongTensor(np.array([obj for i, obj in enumerate(train_y1) if i in rand_index_x1])))

			train_prediction = self.coarse_forward1(batch_x1)

			loss = criterion(train_prediction, batch_y1)

			loss.backward()

			optimizer.step()

			prediction_test = self.coarse_forward1(v_test_x1)
			pre_labels = [Max_Index(line) for line in prediction_test.data.numpy()]
			recall, precision, macrof1, microf1, acc = Get_Report(test_y1, pre_labels)
			print( "[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}    accuracy:{:.4%}".format(epoch, recall, precision, macrof1, microf1, acc))


		v_test_x2 = torch.autograd.Variable(torch.Tensor(np.array([[obj] for obj in test_x2])))

		for epoch in range(200):
			optimizer.zero_grad()

			rand_index_x2 = np.random.choice(len(train_x2), size=32, replace=False)

			batch_x2 = torch.autograd.Variable(torch.Tensor(np.array([[obj] for i, obj in enumerate(train_x2) if i in rand_index_x2])))
			batch_y2 = torch.autograd.Variable( torch.LongTensor(np.array([obj for i, obj in enumerate(train_y2) if i in rand_index_x2])))

			train_prediction = self.coarse_forward2(batch_x2)

			loss = criterion(train_prediction, batch_y2)

			loss.backward()

			optimizer.step()

			prediction_test = self.coarse_forward2(v_test_x2)
			pre_labels = [Max_Index(line) for line in prediction_test.data.numpy()]
			recall, precision, macrof1, microf1, acc = Get_Report(test_y2, pre_labels)
			print("[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}    accuracy:{:.4%}".format(epoch, recall, precision, macrof1, microf1, acc))


		v_test_x3s = torch.autograd.Variable(torch.Tensor(np.array([[obj] for obj in test_x3s])))
		v_test_x3w = torch.autograd.Variable(torch.Tensor(np.array([np.array(obj) for obj in test_x3w])))

		for epoch in range(1000):
			optimizer.zero_grad()

			rand_index_x3 = np.random.choice(len(train_x3s), size=32, replace=False)

			batch_x3s = torch.autograd.Variable(torch.Tensor(np.array([[obj] for i, obj in enumerate(train_x3s) if i in rand_index_x3])))
			batch_x3w = torch.autograd.Variable(torch.Tensor(np.array([obj for i, obj in enumerate(train_x3w) if i in rand_index_x3])))
			batch_y3 = torch.autograd.Variable( torch.LongTensor(np.array([obj for i, obj in enumerate(train_y3) if i in rand_index_x3])))

			train_prediction = self.fine_forward3(batch_x3s, batch_x3w)

			loss = criterion(train_prediction, batch_y3)

			loss.backward()

			optimizer.step()

			prediction_test = self.fine_forward3(v_test_x3s, v_test_x3w)
			pre_labels = [Max_Index(line) for line in prediction_test.data.numpy()]
			recall, precision, macrof1, microf1, acc = Get_Report(test_y3, pre_labels)
			print("[{:4d}]    recall:{:.4%}    precision:{:.4%}    macrof1:{:.4%}    microf1:{:.4%}    accuracy:{:.4%}".format(epoch, recall, precision, macrof1, microf1, acc))
