# Speaker Recognition
Speaker recognition Machine Learning. This engine is a proprietary speaker recognition engine built for Mtg.ai.

The goal is to be able to classify with high accuracy any speaker profile.

## Data

We are using the [VCTK speech corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) to
bootstrap speaker recognition and learn distinguishable features.

The corpus must be extracted into `asset/data` to train the network.

## Architecture

For now the architecture is a simple neural network with 109 output nodes (one for each speaker of 
the speech corpus).

## Training

Execute:

```
python train.py
```

To traing the network.