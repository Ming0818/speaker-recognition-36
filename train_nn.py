from data import VCTK
import sugartensor as tf

batch_size = 16
num_dim = 128
num_blocks = 3

n_nodes_hl1 = 200
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 200

# VCTK corpus input tensor ( with QueueRunner )
data = VCTK(batch_size=batch_size)

x = data.mfcc
y = data.label

logit = (tf.placeholder(tf.float32, shape=(batch_size, 128,))
         .sg_dense(dim=400, act='relu', bn=True)
         .sg_dense(dim=200, act='relu', bn=True)
         .sg_dense(dim=10))

loss = logit.sg_ce(target=y)