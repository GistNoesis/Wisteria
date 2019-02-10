# Wisteria GistNoesis : music tutor using Tensorflow.js

Wisteria website : https://gistnoesis.github.io/

Here is the donate button if you wish to contribute : 

You can contact us at gistnoesis@gmail.com


On this page you will get the backend of Wisteria. 
It's just been created so wait a week for it to get cleaner.

You will learn to generate synthetic datasets, train the models, export them for use inside a browser.

## Requirements : 
-Ubuntu (other OS will need some adaptation)
-ffmpeg
-fluidsynth

Noise downloaded from https://people.xiph.org/~jm/demo/rnnoise/


## Installation :

One virtual env for learning. (Install the needed pip dependencies listed in the file, depending on your machine choose either tensorflow or tensorflow-gpu)

One virtual env for exporting (needed so that pip install tensorflowjs doesn't interfere with your main virtual env)

Inside tfjs folder :
npm install
npm run build 

Generate some ssl certificate (cert.pem and key.pem) using openssl.

(If you want webpack-dev server, you will have to edit the variable host inside index.js to serve the model using an external webserver like http-server -S -p 3000 --cors -C ../cert.pem -K ../key.pem 

## Usage : 
Edit the configs as desired then call from the right virtual env

To generate the dataset and learn the network :
python3 musicTranscriptor.py -ds pianoAllMonophonic10000.datasetconfig -mod conv1.modelconfig -tr robust.trainingconfig -act l

To export the model :
python3 musicTranscriptor.py -ds pianoAllMonophonic10000.datasetconfig -mod conv1.modelconfig -tr robust.trainingconfig -act e


## Some Technical details to consider when designing your networks : 
We are doing some real-time processing therefore it imposes restriction on the keras layers you can use so that everything works well.

When doing filtering (i.e. using only data from the past) and not smoothing (waiting to have more data from the future before making a prediction), you can't use "SAME" or "VALID" convolutions if they have a kernel size greater than 1. You must use "CAUSAL" convolutions. You can use recurrent layers. You can't use normalization layers like BatchNorm or InstanceNorm, so I recommend using selu as a substitute.

When doing smoothing (not yet implemented), you can use an architecture like U-Net segmentation to achieve state of the art.

If you run the network on audio extracted between two specific instants, for example the 5 second following the detection of "Hi Wisteria", then you can use "same" convolution and normalization layers.

We have some more advance specific keras layers dedicated to audio processing in python but they are not ported to js yet. (In part due to a current tensorflowjs issue regarding custom layers, but also to keep the initial release as smooth as possible).

There is still some work to do to model various microphone properly to make it robust to various microphone. Obviously if you can collect your datasets instead of using generated ones then the network may model it from data. For example you can use the MAPS dataset. (We didn't use it)

You can quite easily build some nice audio application following this project.

I recommend working with librispeech dataset and CTC loss for speech recognition, if you want to replicate deepspeech.

Have fun ;)
