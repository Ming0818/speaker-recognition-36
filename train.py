from data import VCTK

batch_size = 16

# VCTK corpus input tensor ( with QueueRunner )
data = VCTK(batch_size=batch_size)

# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = data.mfcc