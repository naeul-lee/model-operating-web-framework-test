import pickle
import numpy as np


def predict_score(clf, X, threshold):
    prob = clf.predict_proba(X)
    prob = prob[:, 1]
    prob[prob >= threshold] = 1
    prob[prob < threshold] = 0
    return prob


data = [[0.01028214, 0.01214058, 0.01282051, 0.08011869, 0.,
         0.017805, 0., 0.09443099, 0.00323415, 0.,
         0., 0., 0., 0.03649635, 0.01666667,
         0.0545977, 0., 0., 0.01754386, 0.0258,
         0.0030303, 0.01941748, 0., 0.0209205, 1.,
         0.00350672, 0.01282051, 0.05586592, 0.0047, 0.,
         0., 0.00578014, 0.07857734, 0., 0.0074528,
         0.05714286, 0.00322928, 0.22151899, 0.20977011, 0.01028214,
         0., 0.08237548, 0.02142857, 0.01898734, 0.01282051,
         0., 1., 0., 0.1, 0]]

data_arr = np.array(data)

model_fname = './sample_model_feature_50_20220712.pkl'

# 모델 불러오기
clf = pickle.load(open(model_fname, 'rb'))

print(predict_score(clf, data_arr, 0.3))
