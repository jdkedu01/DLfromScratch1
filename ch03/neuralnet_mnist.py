# coding: utf-8
import sys, os
#sys.path.append(os.pardir)
sys.path.append(".")  # CursorAI는 각 ch0x에서 수행해도 현재 디렉토리는 "Open한" Root디렉토리로 지정됨
                    # PyCharm은 각 ch0x에서 수행하면 그 .py 파일의 디렉토리가 현재 디렉토리로 지정됨 
sys.path.append("./ch03")                  
current_dir = os.getcwd()  # get current working directory
print("========="+current_dir)
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 현재 파일의 부모의 부모 디렉터리를 추가
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "sample_weight.pkl")
        with open(file_path, 'rb') as f:
            network = pickle.load(f)
        return network
    except FileNotFoundError:
        print("Error: sample_weight.pkl 파일을 찾을 수 없습니다.")
        print("파일을 다음 경로에 다운로드 해주세요:", os.path.dirname(__file__))
        sys.exit(1)

current_dir = os.getcwd()  # get current working directory
print("========="+current_dir)
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
