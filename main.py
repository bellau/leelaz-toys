import tfprocess as tfp
import net_to_model as ntm
import weights
import tensorflow as tf

import os.path
import numpy as np

net_hash = weights.model()
print net_hash
m = ntm.transform(net_hash)

tfprocess = tfp.TFProcess()
tfprocess.init_net(m[0],m[1],m[2])
planes = np.zeros((1, 18, 19, 19))
#planes[0][15][3][3] =  1
planes[0][17] = np.ones((19,19))
y_out, z_out = tfprocess.submit(planes)
print(y_out.shape)
print(np.argmax(y_out, axis=None))
for y in range(19):
    for x in range(19):
        print "(%s,%s) : %s (%s)" % (x+1,y+1,y_out[0][y*19+x], (y*19+x))
print "vic %s " % z_out[0]
