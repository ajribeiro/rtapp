import numpy as np
from sklearn import datasets, linear_model
import cPickle
import matplotlib.pyplot as plt

with open('featurelist.pkl', 'rb') as fid:
    featurelist = cPickle.load(fid)

with open('regval.pkl', 'rb') as fid:
    targets = cPickle.load(fid)

with open('movielist.pkl', 'rb') as fid:
    mlist = cPickle.load(fid)

allx,ally = [],[]
x_train,x_test = [],[]
y_train,y_test = [],[]

for i in range(len(featurelist)):
    f = featurelist[i]
    allx.append([f['actorsum'],f['directorsum'],f['writersum']])
    ally.append(targets[i])
    # for v in f.itervals():

n_trn = int(.8*len(allx))
n_tst = int((len(allx)-n_trn)/2)

x_train = allx[:n_trn]
x_test = allx[n_trn:n_trn+n_tst]
x_val = allx[n_trn+n_tst:]


y_train = ally[:n_trn]
y_test = ally[n_trn:n_trn+n_tst]
y_val = ally[n_trn+n_tst:]


# # # Split the data into training/testing sets
# # diabetes_X_train = diabetes_X_temp[:-20]
# # diabetes_X_test = diabetes_X_temp[-20:]


# # # Split the targets into training/testing sets
# # diabetes_y_train = diabetes.target[:-20]
# # diabetes_y_test = diabetes.target[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(x_train, y_train)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((regr.predict(x_test) - y_test) ** 2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(x_test, y_test))

# rms = 0
# l5,l10,l20 = 0,0,0
# xx,yy = [],[]
# for i in range(len(x_test)):
#     y = regr.decision_function(x_test[i])
#     xx.append(y_test[i])
#     yy.append(y)

#     rms += y_test[i]**2-y**2
#     err = np.abs(y_test[i]-y)

#     if err < 5: l5 += 1
#     if err < 10: l10 += 1
#     if err < 25: l20 += 1

# rms /= float(len(allx))
# rms = np.sqrt(rms)

# print rms

# n = float(len(y_test))
# print l5,l10,l20,len(y_test)
# print l5/n,l10/n,l20/n

# # f = plt.figure()
# # plt.plot(xx,yy,'.b')
# # plt.show()

# f2 = plt.figure()
# # plt.hist(ally)
# # plt.show()

# c = ['r','b','g']
# for i in range(0,3):
#     xx,yy = [],[]
#     for j in range(len(allx)):
#         xx.append(allx[j][i])
#         yy.append(ally[j])
#     plt.plot(xx,yy,'.'+c[i])
#     print np.corrcoef(xx,yy)
# f2.show()



# # # Plot outputs
# # pl.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# # pl.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
# #         linewidth=3)

# # pl.xticks(())
# # pl.yticks(())

# # pl.show()

# from sklearn import svm
# clf = svm.SVR()
# clf.fit(x_train,y_train)
# rms = 0
# l5,l10,l20 = 0,0,0
# xx,yy = [],[]
# for i in range(len(x_test)):
#     y = clf.predict(x_test[i])[0]
#     xx.append(y_test[i])
#     yy.append(y)

#     rms += y_test[i]**2-y**2
#     err = np.abs(y_test[i]-y)

#     if err < 5: l5 += 1
#     if err < 10: l10 += 1
#     if err < 25: l20 += 1

# rms /= float(len(allx))
# rms = np.sqrt(rms)

# print rms

# n = float(len(y_test))
# print l5,l10,l20,len(y_test)
# print l5/n,l10/n,l20/n




# from sklearn import linear_model
# clf = linear_model.Ridge (alpha = .9)
# clf.fit(x_train,y_train)
# rms = 0
# l5,l10,l20 = 0,0,0
# xx,yy = [],[]
# for i in range(len(x_test)):
#     y = clf.decision_function(x_test[i])
#     xx.append(y_test[i])
#     yy.append(y)

#     rms += y_test[i]**2-y**2
#     err = np.abs(y_test[i]-y)

#     if err < 5: l5 += 1
#     if err < 10: l10 += 1
#     if err < 25: l20 += 1

# rms /= float(len(allx))
# rms = np.sqrt(rms)

# print rms

# n = float(len(y_test))
# print l5,l10,l20,len(y_test)
# print l5/n,l10/n,l20/n


# clf = linear_model.Lasso(alpha = 0.1)
# clf.fit(x_train,y_train)
# rms = 0
# l5,l10,l20 = 0,0,0
# xx,yy = [],[]
# for i in range(len(x_test)):
#     y = clf.predict(x_test[i])
#     xx.append(y_test[i])
#     yy.append(y)

#     rms += y_test[i]**2-y**2
#     err = np.abs(y_test[i]-y)

#     if err < 5: l5 += 1
#     if err < 10: l10 += 1
#     if err < 25: l20 += 1

# rms /= float(len(allx))
# rms = np.sqrt(rms)

# print rms

# n = float(len(y_test))
# print l5,l10,l20,len(y_test)
# print l5/n,l10/n,l20/n




from sklearn import neighbors

xx=[]
aa=[]
yy=[]

f = plt.figure()

nn=12
for nn in range(nn,nn+1):
    clf = neighbors.KNeighborsRegressor(nn)
    clf.fit(x_train, y_train)


    rms = 0
    l5,l10,l20 = 0,0,0
    for i in range(len(x_val)):
        y = clf.predict(x_val[i])
        # xx.append(y_test[i])
        # yy.append(y)

        rms += y_val[i]**2-y**2
        err = np.abs(y_val[i]-y)

        if err < 5: l5 += 1
        if err < 10: l10 += 1
        if err < 25: l20 += 1

    rms /= float(len(allx))
    rms = np.sqrt(rms)

    print '****************************'
    print 'n = ',nn
    print rms

    n = float(len(y_val))
    print l5,l10,l20,len(y_val)
    print l5/n,l10/n,l20/n

    xx.append(nn)
    aa.append(l10/n)
    yy.append(l20/n)


with open('knnreg.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

# plt.plot(xx,yy,'-ob')
# plt.plot(xx,aa,'-or')
# f.show()



