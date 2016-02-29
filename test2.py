# _*_ coding: UTF-8 _*_

# µ¼Èësvmutil Ä£¿é
import sys;
sys.path.append('c:/lyj/libsvm-master/python')
from svmutil import *

# Construct problem in python format
y, x = [1, -1], [[1, 0, 1], [-1, 0, -1]]
prob = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
m = svm_train(prob, param)
svm_save_model('value.model', m)
p_label, p_acc, p_val = svm_predict([1,-1],[[1,1,1], [-1,-1,-1]], m)

print(p_label)
