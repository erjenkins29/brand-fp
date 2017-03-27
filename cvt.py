import theano
from theano import sparse
import pandas as pd
import numpy as np

#If shape[0] > shape[1], use csc format. Otherwise, use csr.

filename=['sampledata_qty_amt','sampledata_qty_share','sampledata_rev_amt','sampledata_rev_share']

for i in range(len(filename)):
    df = pd.read_csv('data/sampledata/'+filename[i]+'.csv')
    data = df.iloc[:,2:]
    x = sparse.csc_matrix(name='x',dtype='float64')
    f = theano.function([x],x)
    res = f(data)
    np.save('data/sampledata/sparsecsv/'+filename[i],res)
