import tfprocess as tfp
import net_to_model as ntm
import weights
import tensorflow as tf

import os.path
import numpy as np

net_hash = weights.model()
print net_hash
m = ntm.transform(net_hash)

x = [
    tf.placeholder(tf.float32, [None, 18, 19 * 19])
    #,    
    #tf.placeholder(tf.float32, [None, 362]),
    #tf.placeholder(tf.float32, [None, 1])
    ]


def load_graph(frozen_graph_filename):                                                                                                                     
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:                                                                                                 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())                                                                                                                

    with tf.Graph().as_default() as graph:                                                                                                                 
        tf.import_graph_def(graph_def, name = "")                                                                                                                     

    return graph

tfprocess = tfp.TFProcess()
tfprocess.init_net(x)
if tfprocess.RESIDUAL_BLOCKS != m[1]:
    raise ValueError("Number of blocks in tensorflow model doesn't match "\
            "number of blocks in input network")
if tfprocess.RESIDUAL_FILTERS != m[2]:
    raise ValueError("Number of filters in tensorflow model doesn't match "\
            "number of filters in input network")
tfprocess.replace_weights(m[0])
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