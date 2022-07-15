import pandas as pd
import copy
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# model prob결과에 대한 threshold를 사용하여 최종 prediction결과 전달
def predict_score(clf,X,threshold):
    prob = clf.predict_proba(X)
    prob = prob[:,1]
    prob[prob>= threshold] = 1
    prob[prob< threshold] = 0
    return prob


# 데이터 가져오기
fname = "sample_input_with_50features_scaled.csv"
data = pd.read_csv(fname)

# X,y 설정
X = data[data.columns[1:-1]]
y = copy.deepcopy(data['label'])

# Train/Test 셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100, stratify=y)


# 모델 training
clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.01,n_estimators=200)
clf.fit(X_train, y_train)

# 모델 성능 확인 (Train/Test set)
y_true = y_train
y_pred = clf.predict(X_train)
print("train accuracy:",accuracy_score(y_true,y_pred))

y_true = y_test
y_pred = clf.predict(X_test)
print("test accuracy:",accuracy_score(y_true,y_pred))

# accuracy이외의 성능지료 확인
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"tn:{tn}\nfp:{fp}\nfn:{fn}\ntp:{tp}\nFPR:{fp/(fp+tn)}")

# threshold를 0.5에서 0.3으로 변경하여 확인 하는 결과
print(">>>>>>>with threshold")
threshold = 0.3
y_pred=predict_score(clf,X_test,threshold)
y_true = y_test

print("train accuracy:",accuracy_score(y_true,y_pred))
print("train recall:",recall_score(y_true, y_pred))
print("train precision:",precision_score(y_true, y_pred))
print("train f1 score:",f1_score(y_true, y_pred))



tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"tn:{tn}\nfp:{fp}\nfn:{fn}\ntp:{tp}\nFPR:{fp/(fp+tn)}")

import pickle
model_fname = 'sample_model_feature_50_20220713.pkl'
# 모델 저장
pickle.dump(clf, open(model_fname, 'wb'))