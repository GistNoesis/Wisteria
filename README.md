# Wisteria GistNoesis : music tutor using Tensorflow.js

Wisteria website : https://gistnoesis.github.io/

Here is the donate button if you wish to contribute : [paypal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=KLVTA5CBMGV7J&source=url)

You can contact us at gistnoesis@gmail.com


On this page you will get the backend of Wisteria. 
It's just been created so wait a week for it to get cleaner.

You will learn to generate synthetic datasets, train the models, export them for use inside a browser.

## What's new : 
- We added the python side of the character level voice transcriptor (i.e. audio to text) which uses commons voice dataset (https://voice.mozilla.org/en), to learn more see SpeechToText.md, first tests on a french voice set are promising (though the dataset is small.
- We added Horovod, to use parallelize for the voice transcriptor. The voice transcriptor is independent for the time being but will get factorized soon (i.e. working with config files, export, validation, and sharing the learning code with the music transcriptor)
- We added a transformer architecture, both in python and javascript. You can see the transformer.py example to see how to train on text, or musictranscriptor.py to see how to train for transcription. 
- We added the validation code to check visually the results of the learning in python.
- We added code to check the whether or not a sound font can generate a note of specific height, which will allow us to generate more instruments soon.

## Requirements : 
- Ubuntu (other OS will need some adaptation)

- ffmpeg

- fluidsynth

- Noise downloaded from https://people.xiph.org/~jm/demo/rnnoise/

- Horovod (for voice transcriptor)

## Installation :
(optionnal but recommended if learning) Download the noise and place it in the specified folder (see the README there for more info).

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

To validate the model :
python3 musicTranscriptor.py -ds pianoAllMonophonic10000.datasetconfig -mod conv1.modelconfig -tr robust.trainingconfig -act v
you can also specify a specific dataset config via -vds if you want to validate on a specific dataset, for example the training set to check for overfitting and compare generalization.

## Some Technical details to consider when designing your networks : 
We are doing some real-time processing therefore it imposes restriction on the keras layers you can use so that everything works well.

When doing filtering (i.e. using only data from the past) and not smoothing (waiting to have more data from the future before making a prediction), you can't use "SAME" or "VALID" convolutions if they have a kernel size greater than 1. You must use "CAUSAL" convolutions. You can use recurrent layers. You can use transformer architecture (like GPT-2). You can't use normalization layers like BatchNorm or InstanceNorm, so I recommend using selu or LayerNorm as a substitute.


When doing smoothing (not yet implemented), you can use an architecture like U-Net segmentation to achieve state of the art, alternatively for state of the art you can also use a bidirectionnal transformer architecture (like BERT).

If you run the network on audio extracted between two specific instants, for example the 5 second following the detection of "Hi Wisteria", then you can use "same" convolution and normalization layers.

We have some more advance specific keras layers dedicated to audio processing in python but they are not ported to js yet. (In part due to a current tensorflowjs issue regarding custom layers, but also to keep the initial release as smooth as possible).

There is still some work to do to model various microphone properly to make it robust to various microphone. Obviously if you can collect your datasets instead of using generated ones then the network may model it from data. For example you can use the MAPS dataset. (We didn't use it)

You can quite easily build some nice audio application following this project.

I recommend working with librispeech dataset and CTC loss for speech recognition, if you want to replicate deepspeech.

Have fun ;)
