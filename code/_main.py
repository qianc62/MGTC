import numpy as np
from gensim.models.word2vec import Word2Vec
import torch
from torch import optim
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



all_SC, all_SSR, all_SRL = [], [], []
label_SC, label_SSR, label_SRL = set(), set(), set()

for line in open("./data/MAM-SC.txt").read().split("\n"):
	objs = line.lower().split(", ")
	if len(objs)==2:
		all_SC.append(objs)
		label_SC.add(objs[-1])

for line in open("./data/MAM-SSR.txt").read().split("\n"):
	objs = line.lower().split(", ")
	if line.endswith(", y") and len(objs)==3:
		objs = objs[:-1]
		all_SSR.append(objs)
		label_SSR.add(objs[-1])

for line in open("./data/MAM-SRL.txt").read().split("\n"):
	objs = line.lower().split(", ")
	if len(objs)==3:
		all_SRL.append(objs)
		label_SRL.add(objs[-1])

print(len(all_SC))
print(all_SC[0:10])
print(label_SC)

print(len(all_SSR))
print(all_SSR[0:10])
print(label_SSR)

print(len(all_SRL))
print(all_SRL[0:10])
print(label_SRL)



ratio = 0.80
train_SC,  test_SC  = all_SC[:int(len(all_SC)*ratio)],   all_SC[int(len(all_SC)*ratio):]
train_SSR, test_SSR = all_SSR[:int(len(all_SSR)*ratio)], all_SSR[int(len(all_SSR)*ratio):]
train_SRL, test_SRL = all_SRL[:int(len(all_SRL)*ratio)], all_SRL[int(len(all_SRL)*ratio):]
print(len(train_SC), len(test_SC))
print(len(train_SSR), len(test_SSR))
print(len(train_SRL), len(test_SRL))



w2v_embdding_size = 100
w2v = Word2Vec.load("./w2v/w2v_model")
vocabulary = set(open("./w2v/text8_vocabulary.txt").read().split("\n"))

label_SC  = list(label_SC)
label_SSR = list(label_SSR)
label_SRL = list(label_SRL)


def Encode_Sentence_Data(array, label_map):
	embeddings, labels = [], []
	for line in array:
		words = line[0].split(" ")
		label = line[1]

		mat = []
		for word in words:
			if(word in vocabulary):
				mat.append(w2v[word])
			else:
				mat.append(w2v["a"])
		while len(mat)<10:
			mat.append(w2v["a"])
		mat = mat[:10]

		embeddings.append(mat)
		labels.append(label_map.index(label))

		# print(line)

	return embeddings, labels

def Encode_Word_Data(array, label_map):
	embeddings, wembeddings, labels = [], [], []
	for line in array:
		words = line[0].split(" ")
		label = line[-1]

		mat = []
		for word in words:
			if(word in vocabulary):
				mat.append(w2v[word])
			else:
				mat.append(w2v["a"])
		while len(mat)<10:
			mat.append(w2v["a"])
		mat = mat[:10]

		embeddings.append(mat)

		index = int(line[1])
		center_word = line[0].split(" ")[index]
		if (center_word in vocabulary):
			rep = list(np.array(w2v[center_word]))
			rep.extend([index*1.0])
			rep = [float(obj) for obj in rep]
			wembeddings.append(rep)
		else:
			rep = list(np.array(w2v["a"]))
			rep.extend([index * 1.0])
			rep = [float(obj) for obj in rep]
			wembeddings.append(rep)

		labels.append(label_map.index(label))

		# print(line)

	return embeddings, wembeddings, labels

train_x1, train_y1 = Encode_Sentence_Data(train_SC, label_SC)
test_x1,  test_y1  = Encode_Sentence_Data(test_SC, label_SC)

train_x2, train_y2 = Encode_Sentence_Data(train_SSR, label_SSR)
test_x2,  test_y2  = Encode_Sentence_Data(test_SSR, label_SSR)

train_x3s, train_x3w, train_y3 = Encode_Word_Data(train_SRL, label_SRL)
test_x3s,  test_x3w,  test_y3  = Encode_Word_Data(test_SRL, label_SRL)



class C2F(torch.nn.Module):
	def __init__(self):
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
		self.ST1_fc2 = torch.nn.Linear(50, len(label_SC))

		self.ST2_fc1 = torch.nn.Linear(212 + 50, 50)
		self.ST2_fc2 = torch.nn.Linear(50, len(label_SSR))

		self.ST3_fc1 = torch.nn.Linear(212 + 100 + 50 + 1, 50)
		self.ST3_fc2 = torch.nn.Linear(50, len(label_SRL))

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
		optimizer = optim.Adam(self.parameters(), lr=0.0001)

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

		for epoch in range(300):
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


c2f = C2F()
c2f.train(train_x1, train_y1, test_x1,  test_y1, train_x2, train_y2, test_x2,  test_y2, train_x3s, train_x3w, train_y3, test_x3s,  test_x3w,  test_y3)
