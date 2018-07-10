
# coding: utf-8

# In[46]:


#Run config file
import os
os.system('python Config.py')
os.system('python GlobalsVars.py')


# In[47]:


import sys
import csv
from scipy.io import arff
import numpy as np
import GlobalsVars as v
import glob
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


# In[48]:


print ("Start gs_1 generation")
# folder gs_1
# Feature agglomeration with median window 2s
path = 'gs_1/'
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path + '/arousal/'):
    os.makedirs(path + '/arousal/')
    os.makedirs(path + '/valence/')


# In[49]:


def agglo_fn(X):
    from sklearn.cluster import FeatureAgglomeration
    import pandas as pd
    import matplotlib.pyplot as plt
    if X.shape != (7501,6):
        X = np.transpose(X)
    
    agglo=FeatureAgglomeration(n_clusters=1).fit_transform(X)
    return agglo


# In[50]:


# Median filter
def mfilt(x,size = 2):
    
    from scipy.signal import medfilt
    winmed = medfilt(x,25*size-1)
    
    return winmed
    


# In[58]:


#rating individual files of train and dev on arousal and valence
def listFiles():
    files = []
    #We get a tab countaining each file
    for i, s in enumerate(v.agsi):
        files.append([sorted(filter( lambda f: not f.startswith('.'), os.listdir(s+"."))),s])
    return files;
#End listFiles


# In[59]:


# write generated gold standard gs_1

basedir = path 
header = """@relation GOLDSTANDARD

@attribute Instance_name string
@attribute frameTime numeric
@attribute GoldStandard numeric


@data


"""
files = listFiles()

#X = np.zeros(6,7501)
namespace = globals()
for i in range(len(v.eName)/2):
    for f in files[i][0]:
        #print(files[i][1]+f)
        #print((files[i][1]+f).replace("arousal","valence"))
        df_arr = pd.read_csv(files[i][1]+f,delimiter=';')
        df_val = pd.read_csv((files[i][1]+f).replace("arousal","valence"),delimiter=';')
        fn = os.path.splitext(f)[0]
#         namespace['X_arr_%s' %fn] = df_arr.iloc[:,1:7]
#         namespace['X_val_%s'%fn] = df_val.iloc[:,1:7]
        
        X_arr = np.array((mfilt(df_arr.iloc[:,1]),mfilt(df_arr.iloc[:,2]),                                     mfilt(df_arr.iloc[:,3]),mfilt(df_arr.iloc[:,4]),                                     mfilt(df_arr.iloc[:,5]),mfilt(df_arr.iloc[:,6])))
        X_val = np.array((mfilt(df_val.iloc[:,1]),mfilt(df_val.iloc[:,2]),                                     mfilt(df_val.iloc[:,3]),mfilt(df_val.iloc[:,4]),                                     mfilt(df_val.iloc[:,5]),mfilt(df_val.iloc[:,6])))
        
#         [pca_aro, spca_aro] = spca_fn(X_arr)
#         [pca_val, spca_val] = spca_fn(X_val)
        aggo_aro = agglo_fn(X_arr)
        aggo_val = agglo_fn(X_val)
    
        arff_arr=open(basedir+"arousal/"+f.split(".")[0]+".arff","w")
        arff_arr.write(header)
        for j in range(len(aggo_aro)):
            arff_arr.write("\n"+f.split(".")[0]+","+str(df_arr["time"].iloc[j])+","+str(aggo_aro[j,0]))
        arff_arr.close()
        
        arff_val=open(basedir+"valence/"+f.split(".")[0]+".arff","w")
        arff_val.write(header)
        for j in range(len(aggo_val)):
            arff_val.write("\n"+f.split(".")[0]+","+str(df_val["time"].iloc[j])+","+str(aggo_val[j,0]))
        arff_val.close()


# In[8]:


print ("Start gs_2 generation")
# folder gs_2
# Feature agglomeration with flat window 6s
path = 'gs_2/'
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path + '/arousal/'):
    os.makedirs(path + '/arousal/')
    os.makedirs(path + '/valence/')


# In[9]:


# flat and hanning window
def smooth(x,window_len=25*6,window='flat'):

    import numpy

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
        
    y=numpy.convolve(w/w.sum(),x,mode='same')    
    #y=numpy.convolve(w/w.sum(),s,mode='same')
    return y#[(window_len/2-1):-(window_len/2)]


# In[10]:


# write generated gold standard gs_2

basedir = path 
header = """@relation GOLDSTANDARD

@attribute Instance_name string
@attribute frameTime numeric
@attribute GoldStandard numeric


@data


"""
files = listFiles()

#X = np.zeros(6,7501)
namespace = globals()
for i in range(len(v.eName)/2):
    for f in files[i][0]:
        #print(files[i][1]+f)
        #print((files[i][1]+f).replace("arousal","valence"))
        df_arr = pd.read_csv(files[i][1]+f,delimiter=';')
        df_val = pd.read_csv((files[i][1]+f).replace("arousal","valence"),delimiter=';')
        fn = os.path.splitext(f)[0]
        
        X_arr = np.array((smooth(df_arr.iloc[:,1]),smooth(df_arr.iloc[:,2]),                                     smooth(df_arr.iloc[:,3]),smooth(df_arr.iloc[:,4]),                                     smooth(df_arr.iloc[:,5]),smooth(df_arr.iloc[:,6])))
        X_val = np.array((smooth(df_val.iloc[:,1]),smooth(df_val.iloc[:,2]),                                     smooth(df_val.iloc[:,3]),smooth(df_val.iloc[:,4]),                                     smooth(df_val.iloc[:,5]),smooth(df_val.iloc[:,6]))) 
        
#         [pca_aro, spca_aro] = spca_fn(X_arr)
#         [pca_val, spca_val] = spca_fn(X_val)
        
        
        aggo_aro = agglo_fn(X_arr)
        aggo_val = agglo_fn(X_val)
    
        arff_arr=open(basedir+"arousal/"+f.split(".")[0]+".arff","w")
        arff_arr.write(header)
        for j in range(len(aggo_aro)):
            arff_arr.write("\n"+f.split(".")[0]+","+str(df_arr["time"].iloc[j])+","+str(aggo_aro[j,0]))
        arff_arr.close()
        
        arff_val=open(basedir+"valence/"+f.split(".")[0]+".arff","w")
        arff_val.write(header)
        for j in range(len(aggo_val)):
            arff_val.write("\n"+f.split(".")[0]+","+str(df_val["time"].iloc[j])+","+str(aggo_val[j,0]))
        arff_val.close()


# In[11]:


print ("Start gs_3 generation")
# folder gs_3
# Sparse PCA with median window 2s
path = 'gs_3/'
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path + '/arousal/'):
    os.makedirs(path + '/arousal/')
    os.makedirs(path + '/valence/')


# In[12]:


# Normalization
def mag(X_r):
    from scipy.stats import zscore
    from sklearn.preprocessing import MaxAbsScaler
    max_abs_scaler = MaxAbsScaler()
    z_X_r = zscore(X_r)
    z_X_r = max_abs_scaler.fit_transform(z_X_r)
    return z_X_r


# In[13]:


# Sparse PCA
def spca_fn(X):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.decomposition import SparsePCA
    if X.shape != (7501,6):
        X = np.transpose(X)
    
    pca = PCA(n_components=1)
    X_r = pca.fit(X).transform(X)
    spca = SparsePCA(n_components=1)
    X_r2 = spca.fit(X).transform(X)
    
    return X_r,X_r2
    


# In[14]:


# write generated gold standard gs_3

basedir= path
header="""@relation GOLDSTANDARD

@attribute Instance_name string
@attribute frameTime numeric
@attribute GoldStandard numeric


@data


"""

files = listFiles()

#X = np.zeros(6,7501)
namespace = globals()
for i in range(len(v.eName)/2):
    for f in files[i][0]:
        #print(files[i][1]+f)
        #print((files[i][1]+f).replace("arousal","valence"))
        df_arr = pd.read_csv(files[i][1]+f,delimiter=';')
        df_val = pd.read_csv((files[i][1]+f).replace("arousal","valence"),delimiter=';')
        fn = os.path.splitext(f)[0]
#         namespace['X_arr_%s' %fn] = df_arr.iloc[:,1:7]
#         namespace['X_val_%s'%fn] = df_val.iloc[:,1:7]
        
        X_arr = np.array((mfilt(df_arr.iloc[:,1]),mfilt(df_arr.iloc[:,2]),                                     mfilt(df_arr.iloc[:,3]),mfilt(df_arr.iloc[:,4]),                                     mfilt(df_arr.iloc[:,5]),mfilt(df_arr.iloc[:,6])))
        X_val = np.array((mfilt(df_val.iloc[:,1]),mfilt(df_val.iloc[:,2]),                                     mfilt(df_val.iloc[:,3]),mfilt(df_val.iloc[:,4]),                                     mfilt(df_val.iloc[:,5]),mfilt(df_val.iloc[:,6])))  
        
        [pca_aro, spca_aro] = spca_fn(X_arr)
        [pca_val, spca_val] = spca_fn(X_val)
        pca_aro = mag(pca_aro)
        spca_aro = mag(spca_aro)
        pca_val = mag(pca_val)
        spca_val = mag(spca_val)
    
        arff_arr=open(basedir+"arousal/"+f.split(".")[0]+".arff","w")
        arff_arr.write(header)
        for j in range(len(spca_aro)):
            arff_arr.write("\n"+f.split(".")[0]+","+str(df_arr["time"].iloc[j])+","+str(spca_aro[j,0]))
        arff_arr.close()
        
        arff_val=open(basedir+"valence/"+f.split(".")[0]+".arff","w")
        arff_val.write(header)
        for j in range(len(spca_val)):
            arff_val.write("\n"+f.split(".")[0]+","+str(df_val["time"].iloc[j])+","+str(spca_val[j,0]))
        arff_val.close()


# In[18]:


print ("Start gs_4 generation")
# folder gs_4
# Agglormeration with median window 6s
path = 'gs_4/'
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path + '/arousal/'):
    os.makedirs(path + '/arousal/')
    os.makedirs(path + '/valence/')


# In[ ]:


# Median filter
def mfilt(x,size = 6):
    
    from scipy.signal import medfilt
    winmed = medfilt(x,25*size-1)
    
    return winmed
    


# In[ ]:


# write generated gold standard gs_4

basedir = path 
header = """@relation GOLDSTANDARD

@attribute Instance_name string
@attribute frameTime numeric
@attribute GoldStandard numeric


@data


"""
files = listFiles()

#X = np.zeros(6,7501)
namespace = globals()
for i in range(len(v.eName)/2):
    for f in files[i][0]:
        #print(files[i][1]+f)
        #print((files[i][1]+f).replace("arousal","valence"))
        df_arr = pd.read_csv(files[i][1]+f,delimiter=';')
        df_val = pd.read_csv((files[i][1]+f).replace("arousal","valence"),delimiter=';')
        fn = os.path.splitext(f)[0]
#         namespace['X_arr_%s' %fn] = df_arr.iloc[:,1:7]
#         namespace['X_val_%s'%fn] = df_val.iloc[:,1:7]
        
        X_arr = np.array((mfilt(df_arr.iloc[:,1]),mfilt(df_arr.iloc[:,2]),                                     mfilt(df_arr.iloc[:,3]),mfilt(df_arr.iloc[:,4]),                                     mfilt(df_arr.iloc[:,5]),mfilt(df_arr.iloc[:,6])))
        X_val = np.array((mfilt(df_val.iloc[:,1]),mfilt(df_val.iloc[:,2]),                                     mfilt(df_val.iloc[:,3]),mfilt(df_val.iloc[:,4]),                                     mfilt(df_val.iloc[:,5]),mfilt(df_val.iloc[:,6])))
        
#         [pca_aro, spca_aro] = spca_fn(X_arr)
#         [pca_val, spca_val] = spca_fn(X_val)
        aggo_aro = agglo_fn(X_arr)
        aggo_val = agglo_fn(X_val)
    
        arff_arr=open(basedir+"arousal/"+f.split(".")[0]+".arff","w")
        arff_arr.write(header)
        for j in range(len(aggo_aro)):
            arff_arr.write("\n"+f.split(".")[0]+","+str(df_arr["time"].iloc[j])+","+str(aggo_aro[j,0]))
        arff_arr.close()
        
        arff_val=open(basedir+"valence/"+f.split(".")[0]+".arff","w")
        arff_val.write(header)
        for j in range(len(aggo_val)):
            arff_val.write("\n"+f.split(".")[0]+","+str(df_val["time"].iloc[j])+","+str(aggo_val[j,0]))
        arff_val.close()


# In[79]:


print ("Start gs_5 generation")
# folder gs_5
# with distribution regularization
path = 'gs_5/'
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path + '/arousal/'):
    os.makedirs(path + '/arousal/')
    os.makedirs(path + '/valence/')


# In[81]:


# Training distribution
arousal_gd_train_path = 'pre_train/arousal/train*.arff'
arousal_gd_dev_path = 'pre_train/arousal/dev*.arff'
valence_gd_train_path = 'pre_train/valence/train*.arff'
valence_gd_dev_path = 'pre_train/valence/dev*.arff'

agdtrfname = sorted(glob.glob(arousal_gd_train_path))
arousal_gd_train = []
for f in agdtrfname:
    temp = pd.read_csv(f, sep=';')
    temp = temp['@relation GOLDSTANDARD_AROUSAL'].apply(lambda x: x.split(',')[-1])
    temp = temp[4:]
    temp = pd.to_numeric(temp)
    #print(f)
    arousal_gd_train.extend(temp)
    
vgdtrfname = sorted(glob.glob(valence_gd_train_path))
valence_gd_train = []
for fv in vgdtrfname:
    tempv = pd.read_csv(fv, sep=';')
    tempv = tempv['@relation GOLDSTANDARD_VALENCE'].apply(lambda x: x.split(',')[-1])
    tempv = tempv[4:]
    tempv = pd.to_numeric(tempv)
    #print(fv)
    valence_gd_train.extend(tempv) 

agdevfname = sorted(glob.glob(arousal_gd_dev_path))
arousal_gd_dev = []
for fd in agdevfname:
    temp = pd.read_csv(fd, sep=';')
    temp = temp['@relation GOLDSTANDARD_AROUSAL'].apply(lambda x: x.split(',')[-1])
    temp = temp[4:]
    temp = pd.to_numeric(temp)
    #print(fd)
    arousal_gd_dev.extend(temp)
    
vgdevfname = sorted(glob.glob(valence_gd_dev_path))
valence_gd_dev = []
for fdv in vgdevfname:
    tempv = pd.read_csv(fdv, sep=';')
    tempv = tempv['@relation GOLDSTANDARD_VALENCE'].apply(lambda x: x.split(',')[-1])
    tempv = tempv[4:]
    tempv = pd.to_numeric(tempv)
    #print(fdv)
    valence_gd_dev.extend(tempv) 


# In[82]:


# Load data
val_gd_data = valence_gd_train
aro_gd_data = arousal_gd_train

del valence_gd_train,arousal_gd_train

X_train = np.array([val_gd_data,aro_gd_data])
X_train = np.transpose(X_train)
  
X_test = np.array([valence_gd_dev,arousal_gd_dev])
X_test = np.transpose(X_test)

# Generate some abnormal novel observations
# ran = np.random.uniform(low=-1, high=1, size=(1000, 1))
# aro_out = np.setdiff1d(ran, aro_gd_data)
# aro_out = np.setdiff1d(aro_out, aro_gd_dev_data)
# val_out = np.setdiff1d(ran, val_gd_data)
# val_out = np.setdiff1d(val_out, val_gd_dev_data)
# aro_outliers = np.random.choice(aro_out, 50, replace=False)
# val_outliers = np.random.choice(val_out, 50, replace=False)

# X_outliers = np.array([val_outliers,aro_outliers])
# X_outliers = np.transpose(X_outliers)




# In[83]:


# Construct Regularization
X = np.r_[X_train,X_test]
n_neighbors=50


# In[138]:


def regulof(X):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=50)
    y = clf.fit_predict(X)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
    Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return y


# In[114]:


gs = X
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=50)
neigh.fit(gs) 
NearestNeighbors(algorithm='auto', leaf_size=30)
A  = neigh.kneighbors_graph(gs)


distances, indices = neigh.kneighbors(gs)

# 1 for inlier, -1 for outlier


# In[141]:


# plot the level sets of the decision function


# In[146]:


#rating individual files of train and dev on arousal and valence
def listFilesds(basedir):
    files = []
    gs5 = [basedir +'/arousal/', basedir +'/valence/']    
    #We get a tab countaining each file
    for i, s in enumerate(gs5):
        files.append([sorted(filter( lambda f: not f.startswith('.'), os.listdir(s+"."))),s])
    return files;
#End listFiles


# In[153]:


# write generated gold standard gs_5

readdir ='gs_4' 
basedir = path 
header = """@relation GOLDSTANDARD

@attribute Instance_name string
@attribute frameTime numeric
@attribute GoldStandard numeric


@data


"""
files = listFilesds(readdir)

#X = np.zeros(6,7501)
namespace = globals()
for i in range(len(v.eName)/2):
    for f in files[i][0]:
        #print(files[i][1]+f)
        #print((files[i][1]+f).replace("arousal","valence"))
        tempr = pd.read_csv(files[i][1]+f,delimiter=';')
        tempr = tempr['@relation GOLDSTANDARD'].apply(lambda x: x.split(',')[-1])
        tempr = tempr[4:]
        df_arr = pd.to_numeric(tempr)
        
        tempv = pd.read_csv((files[i][1]+f).replace("arousal","valence"),delimiter=';')
        tempv = tempv['@relation GOLDSTANDARD'].apply(lambda x: x.split(',')[-1])
        tempv = tempv[4:]
        df_val = pd.to_numeric(tempv)
        
        fn = os.path.splitext(f)[0]
        
        X_temp = np.array([df_val,df_arr])
        X_temp = np.transpose(X_temp)
        X_re = np.r_[X,X_temp]
        y = regulof(X_re)
        
#         X_arr = np.array((mfilt(df_arr.iloc[:,1]),mfilt(df_arr.iloc[:,2]),\
#                                      mfilt(df_arr.iloc[:,3]),mfilt(df_arr.iloc[:,4]),\
#                                      mfilt(df_arr.iloc[:,5]),mfilt(df_arr.iloc[:,6])))
#         X_val = np.array((mfilt(df_val.iloc[:,1]),mfilt(df_val.iloc[:,2]),\
#                                      mfilt(df_val.iloc[:,3]),mfilt(df_val.iloc[:,4]),\
#                                      mfilt(df_val.iloc[:,5]),mfilt(df_val.iloc[:,6])))
        
#         [pca_aro, spca_aro] = spca_fn(X_arr)
#         [pca_val, spca_val] = spca_fn(X_val)
#         aggo_aro = agglo_fn(X_arr)
#         aggo_val = agglo_fn(X_val)
    
#         arff_arr=open(basedir+"arousal/"+f.split(".")[0]+".arff","w")
#         arff_arr.write(header)
#         for j in range(len(df_arr)):
#             arff_arr.write("\n"+f.split(".")[0]+","+str(tempr["time"].iloc[j])+","+str(X_temp[j,0]))
#         arff_arr.close()
        
#         arff_val=open(basedir+"valence/"+f.split(".")[0]+".arff","w")
#         arff_val.write(header)
#         for j in range(len(df_val)):
#             arff_val.write("\n"+f.split(".")[0]+","+str(df_val["time"].iloc[j])+","+str(X_temp[j,1]))
#         arff_val.close()


# In[ ]:


# flat and hanning window
def smooth(x,window_len=25*6,window='hanning'):

    import numpy

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
        
    y=numpy.convolve(w/w.sum(),x,mode='same')    
    #y=numpy.convolve(w/w.sum(),s,mode='same')
    return y#[(window_len/2-1):-(window_len/2)]


# In[135]:


# write generated gold standard gs_5

basedir = path 
header = """@relation GOLDSTANDARD

@attribute Instance_name string
@attribute frameTime numeric
@attribute GoldStandard numeric


@data


"""
files = listFiles()

#X = np.zeros(6,7501)
namespace = globals()
for i in range(len(v.eName)/2):
    for f in files[i][0]:
        #print(files[i][1]+f)
        #print((files[i][1]+f).replace("arousal","valence"))
        df_arr = pd.read_csv(files[i][1]+f,delimiter=';')
        df_val = pd.read_csv((files[i][1]+f).replace("arousal","valence"),delimiter=';')
        fn = os.path.splitext(f)[0]
#         namespace['X_arr_%s' %fn] = df_arr.iloc[:,1:7]
#         namespace['X_val_%s'%fn] = df_val.iloc[:,1:7]
        
#         X_arr = np.array((mfilt(df_arr.iloc[:,1]),mfilt(df_arr.iloc[:,2]),\
#                                      mfilt(df_arr.iloc[:,3]),mfilt(df_arr.iloc[:,4]),\
#                                      mfilt(df_arr.iloc[:,5]),mfilt(df_arr.iloc[:,6])))
#         X_val = np.array((mfilt(df_val.iloc[:,1]),mfilt(df_val.iloc[:,2]),\
#                                      mfilt(df_val.iloc[:,3]),mfilt(df_val.iloc[:,4]),\
#                                      mfilt(df_val.iloc[:,5]),mfilt(df_val.iloc[:,6])))
        
#         [pca_aro, spca_aro] = spca_fn(X_arr)
#         [pca_val, spca_val] = spca_fn(X_val)
        aggo_aro = agglo_fn(X_arr)
        aggo_val = agglo_fn(X_val)
    
        arff_arr=open(basedir+"arousal/"+f.split(".")[0]+".arff","w")
        arff_arr.write(header)
        for j in range(len(aggo_aro)):
            arff_arr.write("\n"+f.split(".")[0]+","+str(df_arr["time"].iloc[j])+","+str(aggo_aro[j,0]))
        arff_arr.close()
        
        arff_val=open(basedir+"valence/"+f.split(".")[0]+".arff","w")
        arff_val.write(header)
        for j in range(len(aggo_val)):
            arff_val.write("\n"+f.split(".")[0]+","+str(df_val["time"].iloc[j])+","+str(aggo_val[j,0]))
        arff_val.close()

