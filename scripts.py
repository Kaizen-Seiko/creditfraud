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

""" Không có cột nào có null values 
Nhận thấy có V28 -> trong dữ liệu có 28 features là các phiên bản biến đổi của PCA,
nhưng Amount là giá trị gốc. -> check thử min, max Amount xem sao
"""

# print(min(data.Amount), max(data.Amount))   # 0.0 25691.16
""" Số tiền chi max - min có biên độ khá lớn, có thể làm sai lệch kết quả.
-> các giá trị bị phân tán cao -> dùng StandardScaler() chỉnh lại các biến để có SD = 1"""

sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

""" Trong dataset có biến Time, là yếu tố quyết định bên ngoài, trong quá trình lập 1 mô hình
ta có thể loại bỏ nó"""

# print(data.shape)   # (284807, 31)

""" Bỏ cột Time -> 31 col -> 30 col"""
data.drop(['Time'], axis=1, inplace=True)

""" Ta cũng có thể kiểm tra transcation bị dup. Trước khi remove bất kỳ duplicate trans nào thì check
lại xem đang có bao nhiêu observation, sau đó remove và check lại"""

# print(data.shape)   # (284807, 30)

data.drop_duplicates(inplace=True)

# print(data.shape)   # (275663, 30)

""" Có đâu có 9144 transc bị dup
 Sau khi bỏ các yếu tố thừa, dup ta có 1 dataset đc scale phù hợp -> build model"""

""" Train & Test Split
Xác định biến phụ thuộc X và biến độ lập y"""

X = data.drop('Class', axis=1).values
y = data['Class'].values

""" split our train & test data
see more: https://www.w3schools.com/python/python_ml_train_test.asp
Đại khái: chia dataset ban đầu thành 2 dataset: train và test (tùy tỉ lệ, thường 80-20)
Dữ liệu train đc dùng để training model, còn y là test dùng để testing model
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

""" Model Building 
- Thử từng ML models thì sẽ dễ dàng chọn model phù hợp
- Có thể thay đổi các parameter để tối ưu hóa mô hình. Nhưng nếu kết quả đã chính xác với ít parameter 
thì ko cần làm cho nó phức tạp thêm
"""

""" Decision Tree """
# DT = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# DT.fit(X_train, y_train)
# dt_yhat = DT.predict(X_test)

""" Kiểm tra độ chính xác của Mô hình Decision Tree"""
# print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, dt_yhat)))

""" Check F1-Score of Mô hình Decision Tree
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

""" Check accuracy của K-Nearest Neighbor model"""

# print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))

""" Check F1-Score của K-Nearest Neighbors model"""

# print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))

""" Logistic Regression"""

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# lr_yhat = lr.predict(X_test)

""" Check accuracy của Logistic Regression model"""

# print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))

""" Check F1-Score của Logistic Regression model"""

# print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))

""" Support Vector Machines"""
# svm = SVC()
# svm.fit(X_train, y_train)
# svm_yhat = svm.predict(X_test)

""" Check accuracy của Support Vector Machines model"""
# print('Accuracy score of the Support Vector Machines model is {}'.format(accuracy_score(y_test, svm_yhat)))

""" Check F1-Score của Support Vector Machines model"""
# print('F1 score of the Support Vector Machines model is {}'.format(f1_score(y_test, svm_yhat)))

""" Random Forest"""
# rf = RandomForestClassifier(max_depth = 4)
# rf.fit(X_train, y_train)
# rf_yhat = rf.predict(X_test)

""" Check accuracy của Random Forest model"""
# print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))

""" Check F1-Score của Random Forest model"""
# print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))

""" XGBoost"""
xgb = XGBClassifier(max_depth=4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

""" Check accuracy của XGBoost model"""
print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))

""" Check F1-Score của XGBoost model"""
print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))


""" CONCLUSION
- XGBoost đang 99.95% accuracy
"""

