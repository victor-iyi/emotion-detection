# emotion-detection

Classification of human emotions with a deep convolutional neural network.


## Table of coontents

* [Emotion classes](#emotion-classes)
* [Requirements](#requirements)
  * [Postman chrome extension](#postman-chrome-extension)
* [Usage](#usage)
* [Credits](#credits)

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

## Usage

To run open terminal or command prompt and type in the following commands

```bash
python3 __init__.py
```

After that, open up your browser and go to `https://localhost:5000/api`. You'll see a `json` response. But you won't be able to send a `POST` request through your browser.

### Postman chrome extension

To do this you'll need to install a chrome extension called **Postman**. Just google _postman chrome extension_ and install it.

After installation open it and you're good to go.

## Credits

Sam and Victor I. Afolabi

