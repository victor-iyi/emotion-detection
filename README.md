# emotion-detection

Classification of human emotions with a deep convolutional neural network.

## Emotion classes

* centerlight
* glasses
* leftlight
* noglasses
* normal
* rightlight
* sad
* sleepy
* surprised
* wink

The network consists of a five layer convolutional neural network with 3 fully connected layer and a regression/output layer

## Requirements

In order to run the code. You need to `pip install` the following:

```bash
pip install <package-name>
```

* numpy
* pandas
* flask
* tensorflow
* PIL or Pillow
* tqdm

Tensorflow's installation could be a bit over-head. If that's the case with you, visist the [tensorflow docs](https://www.tensorflow.org/install/) for detailed installation process for your operating system.

