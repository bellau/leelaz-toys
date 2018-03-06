import tfprocess as tfp
import weights
import numpy as np

m = weights.model()

tfprocess = tfp.TFProcess()
tfprocess.init_net(m[0],m[1],m[2])
planes = np.zeros((1, 18, 19, 19))
#planes[0][15][3][3] =  1
planes[0][17] = np.ones((19,19))
y_out, z_out = tfprocess.submit(planes)
print "vic %s " % z_out[0]
index = np.argmax(y_out, axis=None)
print "best move (%s,%s)" % ( int(index / 19) + 1 , (index % 19) +1 )