# _*_ coding: UTF-8 _*_

# 导入svmutil模块
import sys;
sys.path.append('c:/lyj/libsvm-master/python')
from svmutil import *

# 载入heart_scale数据
y, x = svm_read_problem('c:/lyj/libsvm-master/heart_scale')
# LIBSVM 训练
model = svm_train(y[:200], x[:200], '-c 5')
# 保存训练好的model
svm_save_model('heart_scale.model', model)
# LIBSVM 预测
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], model)

print(p_label)
print(p_acc) 
print(p_val)
