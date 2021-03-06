# Packages related to general operating system & warnings
import warnings

warnings.filterwarnings('ignore')
# Packages related to data importing, manipulation, exploratory data #analysis, data understanding
# import numpy as np
import pandas as pd
# from pandas import Series, DataFrame
# Packages related to data visualizaiton
# import seaborn as sns
# %matplotlib inline
# import matplotlib.pyplot as plt

# Setting plot sizes and type of plot
# plt.rc("font", size=14)
# plt.rcParams['axes.grid'] = True
# plt.figure(figsize=(6, 3))
# plt.gray()
# from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.impute import MissingIndicator, SimpleImputer
# from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
# import statsmodels.formula.api as smf
# import statsmodels.tsa as tsa
# from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# from matplotlib import _api, animation, cbook
# from PIL import Image

""" Import dataset"""
data = pd.read_csv("creditcard.csv")

""" Print some leading rows in dataset"""
# print(data.head())

""" Filter normal vs. fraud transaction by Class col"""
Total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent / normal * 100, 2)
# print(cl('Total number of Transactions are {}'.format(Total_transactions), attrs=['bold']))
# print(cl('Number of Normal Transactions are {}'.format(normal), attrs=['bold']))
# print(cl('Number of fraudulent Transactions are {}'.format(fraudulent), attrs=['bold']))
# print(cl('Percentage of fraud Transactions is {}'.format(fraud_percentage), attrs=['bold']))

""" Check null values """
# data.info()

""" Kh??ng c?? c???t n??o c?? null values 
Nh???n th???y c?? V28 -> trong d??? li???u c?? 28 features l?? c??c phi??n b???n bi???n ?????i c???a PCA,
nh??ng Amount l?? gi?? tr??? g???c. -> check th??? min, max Amount xem sao
"""

# print(min(data.Amount), max(data.Amount))   # 0.0 25691.16
""" S??? ti???n chi max - min c?? bi??n ????? kh?? l???n, c?? th??? l??m sai l???ch k???t qu???.
-> c??c gi?? tr??? b??? ph??n t??n cao -> d??ng StandardScaler() ch???nh l???i c??c bi???n ????? c?? SD = 1"""

sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

""" Trong dataset c?? bi???n Time, l?? y???u t??? quy???t ?????nh b??n ngo??i, trong qu?? tr??nh l???p 1 m?? h??nh
ta c?? th??? lo???i b??? n??"""

# print(data.shape)   # (284807, 31)

""" B??? c???t Time -> 31 col -> 30 col"""
data.drop(['Time'], axis=1, inplace=True)

""" Ta c??ng c?? th??? ki???m tra transcation b??? dup. Tr?????c khi remove b???t k??? duplicate trans n??o th?? check
l???i xem ??ang c?? bao nhi??u observation, sau ???? remove v?? check l???i"""

# print(data.shape)   # (284807, 30)

data.drop_duplicates(inplace=True)

# print(data.shape)   # (275663, 30)

""" C?? ????u c?? 9144 transc b??? dup
 Sau khi b??? c??c y???u t??? th???a, dup ta c?? 1 dataset ??c scale ph?? h???p -> build model"""

""" Train & Test Split
X??c ?????nh bi???n ph??? thu???c X v?? bi???n ????? l???p y"""

X = data.drop('Class', axis=1).values
y = data['Class'].values

""" split our train & test data
see more: https://www.w3schools.com/python/python_ml_train_test.asp
?????i kh??i: chia dataset ban ?????u th??nh 2 dataset: train v?? test (t??y t??? l???, th?????ng 80-20)
D??? li???u train ??c d??ng ????? training model, c??n y l?? test d??ng ????? testing model
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

""" Model Building 
- Th??? t???ng ML models th?? s??? d??? d??ng ch???n model ph?? h???p
- C?? th??? thay ?????i c??c parameter ????? t???i ??u h??a m?? h??nh. Nh??ng n???u k???t qu??? ???? ch??nh x??c v???i ??t parameter 
th?? ko c???n l??m cho n?? ph???c t???p th??m
"""

""" Decision Tree """
# DT = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# DT.fit(X_train, y_train)
# dt_yhat = DT.predict(X_test)

""" Ki???m tra ????? ch??nh x??c c???a M?? h??nh Decision Tree"""
# print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, dt_yhat)))

""" Check F1-Score of M?? h??nh Decision Tree
see more: https://www.ritchieng.com/machinelearning-f1-score/
"""
# print('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, dt_yhat)))

""" Check confusion matrix: 
see more: https://machinelearningmastery.com/confusion-matrix-machine-learning/
"""

# print(confusion_matrix(y_test, dt_yhat, labels = [0, 1]))

""" K-Nearest Neighbors model"""
# n = 7
# KNN = KNeighborsClassifier(n_neighbors = n)
# KNN.fit(X_train, y_train)
# knn_yhat = KNN.predict(X_test)

""" Check accuracy c???a K-Nearest Neighbor model"""

# print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))

""" Check F1-Score c???a K-Nearest Neighbors model"""

# print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))

""" Logistic Regression"""

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# lr_yhat = lr.predict(X_test)

""" Check accuracy c???a Logistic Regression model"""

# print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))

""" Check F1-Score c???a Logistic Regression model"""

# print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))

""" Support Vector Machines"""
# svm = SVC()
# svm.fit(X_train, y_train)
# svm_yhat = svm.predict(X_test)

""" Check accuracy c???a Support Vector Machines model"""
# print('Accuracy score of the Support Vector Machines model is {}'.format(accuracy_score(y_test, svm_yhat)))

""" Check F1-Score c???a Support Vector Machines model"""
# print('F1 score of the Support Vector Machines model is {}'.format(f1_score(y_test, svm_yhat)))

""" Random Forest"""
# rf = RandomForestClassifier(max_depth = 4)
# rf.fit(X_train, y_train)
# rf_yhat = rf.predict(X_test)

""" Check accuracy c???a Random Forest model"""
# print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))

""" Check F1-Score c???a Random Forest model"""
# print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))

""" XGBoost"""
xgb = XGBClassifier(max_depth=4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

""" Check accuracy c???a XGBoost model"""
print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))

""" Check F1-Score c???a XGBoost model"""
print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))


""" CONCLUSION
- XGBoost ??ang 99.95% accuracy
"""

