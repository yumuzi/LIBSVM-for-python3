# _*_ coding: UTF-8 _*_

# ����svmutilģ��
import sys;
sys.path.append('c:/lyj/libsvm-master/python')
from svmutil import *

# ����heart_scale����
y, x = svm_read_problem('c:/lyj/libsvm-master/heart_scale')
# LIBSVM ѵ��
model = svm_train(y[:200], x[:200], '-c 5')
# ����ѵ���õ�model
svm_save_model('heart_scale.model', model)
# LIBSVM Ԥ��
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], model)

print(p_label)
print(p_acc) 
print(p_val)
