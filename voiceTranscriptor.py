import numpy as np
import tensorflow as tf
import os
from subprocess import call

from musicTranscriptor import make_dir, resBlockKerasConv1DSelu, make_noise_iterator
import glob
import csv
from musicTranscriptor import _bytes_feature
from musicTranscriptor import read_wav_file,spectogram
import unidecode

from transformer import LayerPositionEmbedding, transformerBlock,transformerBlockGelu
import horovod.tensorflow as hvd

def convertMp3s(language):
    donepath = "datasets/commonvoice/"+language+"/wav/done.txt"
    if (os.path.isfile(donepath)):
        return
    mp3s = glob.glob("datasets/commonvoice/"+ language + "/clips/*.mp3")
    if( len(mp3s) == 0):
        return

    for path in mp3s:
        basename = os.path.splitext( os.path.basename(path) )[0]
        dir = "datasets/commonvoice/"+language+"/wav/" + basename[0:3] + "/" + basename[3:6]
        make_dir( dir )
        wavname = dir + "/" + basename + ".wav"
        call(["ffmpeg", "-i", path, "-ar", "16000", wavname])


    make_dir("datasets/commonvoice/"+language+"/wav/")
    open(donepath, 'a').close()
    return None

def buildDatasetCommonVoiceIfNotExist( language, type, maxaudioLength, maxTextLength ):
    stringname = language + "_" + type
    filename = "datasets/commonvoice/" + stringname + ".tfrecords"
    if (os.path.isfile(filename)):
        return

    convertMp3s(language)


    with tf.python_io.TFRecordWriter(filename) as writer:
        with open("datasets/commonvoice/"+language + "/" + type +".tsv", "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if( i == 0 ):
                    #skipping the header
                    continue
                path = line[1]
                sentence = line[2]
                wave = read_wav_file( "datasets/commonvoice/"+language+"/wav/" + path[0:3]+"/" + path[3:6] +"/" + path + ".wav")
                audioLength = wave.shape[0]
                print("sample : " + str(i) )
                print("audio length : " + str(audioLength))

                if( len(sentence) <= maxTextLength and audioLength <= maxaudioLength ):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'sentence': _bytes_feature(tf.compat.as_bytes(sentence)),
                                'wavfile_path': _bytes_feature(tf.compat.as_bytes(path))
                            }))
                    writer.write(example.SerializeToString())

    return None


def computePathFromName( language, name ):
    language = language.decode("utf-8")
    name = name.decode("utf-8")
    #print(language)
    #print(name)
    path = "".join(["datasets/commonvoice/",language,"/wav/",name[0:3],"/", name[3:6],"/", name,".wav"])
    return path

def removeSpecialChars( sentence ):
    usentence = sentence.decode("utf-8")
    return unidecode.unidecode(usentence)

def toCharList( sentence, numletters ):
    out = np.array( [ c for c in sentence],dtype=np.int32)
    out = np.clip(out,0,numletters)
    return out



def make_ctc_iterator(bs,language, type, vocSize):
    stringname = language + "_" + type
    filename = "datasets/commonvoice/" + stringname + ".tfrecords"
    dataset = tf.data.TFRecordDataset([filename])

    def record_parser(record):
        features = {"sentence": tf.FixedLenFeature((), tf.string, default_value=""),
                    "wavfile_path": tf.FixedLenFeature((), tf.string, default_value=""),
                    }
        parsed_features = tf.parse_single_example(record, features)
        sentence = parsed_features["sentence"]



        wavfile_path = parsed_features["wavfile_path"]

        #lang = tf.constant( language,dtype=tf.string)
        sentence =  tf.py_func(removeSpecialChars,[sentence],tf.string)
        #chars = tf.string_split(sentence, delimiter="")
        chars = tf.py_func(toCharList,[sentence,vocSize],tf.int32)
        #pathwav = tf.strings.join(["datasets/commonvoice/",language,"/wav/",wavfile_path[0:3],"/", wavfile_path[3:6],"/", wavfile_path,".wav"])
        pathwav = tf.py_func(computePathFromName,[language,tf.cast(wavfile_path,tf.string)],tf.string)
        wavfile = tf.py_func(read_wav_file, [pathwav], tf.float32)
        seqlen = tf.expand_dims( tf.shape(wavfile)[0],axis=0)
        #wavfile = tf.squeeze(wavfile, axis=0)
        # wavfile = tf.Print(wavfile,[tf.shape(wavfile)],"tfshapewavfile",summarize=5)
        # img = tf.io.decode_raw( parsed_features["image_data"],tf.float32 )

        # img = tf.io.decode_raw( )
        return [wavfile,chars,seqlen]

    dataset = dataset.shuffle(buffer_size=1000000)
    dataset = dataset.map(record_parser, num_parallel_calls=10)
    dataset = dataset.padded_batch(bs,padded_shapes=( [None], [None], 1 ), drop_remainder=True)
    dataset = dataset.prefetch(10 * bs)
    # dataset = dataset.repeat(1)
    iterator = dataset.make_initializable_iterator()
    return iterator



    return None



def modelTransformer( depth,layerSize,localAttention):
    kernelSize = 1
    nbhead = 4
    kdim = 40
    outSize = 128
    logspecter = tf.keras.Input(shape=(None, 513), dtype=tf.float32, name='logspectrum')
    h = tf.keras.layers.Conv1D(layerSize, kernelSize)(logspecter)
    nbpast = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

    # add past length to position embedding
    pos = LayerPositionEmbedding( int( layerSize // 2))([h, nbpast])
    h = tf.keras.layers.Add()([pos, h])

    pastKs = []
    pastVs = []
    presentKs = []
    presentVs = []


    for i in range(depth):
        h = transformerBlockGelu(h,localAttention, nbhead, kdim, int(layerSize // nbhead), pastKs, pastVs, presentKs, presentVs)


    logit = tf.keras.layers.Conv1D(outSize, 1)(h)

    model = tf.keras.Model(inputs=[logspecter, nbpast] + pastKs + pastVs, outputs=[logit] + presentKs + presentVs)
    return model

def modelConv1( depth,layerSize,kernelSize):
    #kernelSize = 1
    outSize = 128

    logspecter = tf.keras.Input(shape=(None, 513), dtype=tf.float32, name='logspectrum')
    # h = logspecter
    h = tf.keras.layers.Conv1D(layerSize, kernelSize, padding="SAME")(logspecter)
    for i in range(depth):
        h = resBlockKerasConv1DSelu(h, layerSize, kernelSize, 1)

    h = tf.keras.layers.Conv1D(layerSize, kernelSize, activation=tf.keras.activations.selu, padding="SAME",
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                                        distribution="normal"))(h)

    out = tf.keras.layers.Conv1D(outSize, kernelSize, padding="SAME")(h)
    # out = tf.keras.layers.Conv1D(outSize, kernelSize, padding="SAME")( logspecter)
    model = tf.keras.Model(inputs=[logspecter], outputs=[out])
    return model

def buildModel(localattention):
    return modelTransformer(6,300,localattention)
    #return modelConv1(4,100,5)


def removeDoublons(tab):
    out = []
    cur = -1
    for i in range(len(tab)):
        if(tab[i] != cur):
            cur = tab[i]
            out.append(cur)

    #print("removedoublones")
    #print(out)
    return out


def learn():
    bs = 64
    vocSize = 128
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    causalWindow = 50
    noise = True

    itr = make_ctc_iterator(bs,"fr","train",vocSize)
    w,s,l = itr.get_next()

    sparsetarget = tf.contrib.layers.dense_to_sparse(s, eos_token=0)

    if ( noise is True):
        noise_itr = make_noise_iterator(bs, 160000)
        noisedata = noise_itr.get_next()
        w = w + tf.random.uniform((bs, 1), 0.0, 4.0) * noisedata[:,0:tf.shape(w)[1]]


    fft = spectogram(w)
    power = tf.real(fft * tf.math.conj(fft))
    power = tf.reshape(power, (bs, -1, 513))

    eps = 1e-10
    logpower = tf.log(eps + power)

    #h = tf.layers.Conv1D(100,5,padding="SAME",activation=tf.nn.selu)(logpower)
    #predlogit = tf.layers.Conv1D(vocSize,5,padding="SAME")(h)
    model = buildModel(causalWindow)

    modelInputs = [logpower]

    #modelInputs.append(tf.random.uniform((bs, 1), minval=0, maxval=10000, dtype=tf.int32))
    modelInputs.append(tf.zeros((bs, 1), dtype=tf.int32))


    # a causal Window of size 10, use exactly 10 times, so the first present need only 9 pasts, hence the minus one.
    # the edge case localAttention = 0, mean that we will use all pasts, which here in learning mean 0
    localAttention = max(causalWindow - 1, 0)
    for i in range(len(model.inputs) - 2):
        inp = model.inputs[i + 2]
        shape = (bs, inp.shape[1], localAttention, inp.shape[3])
        modelInputs.append(tf.zeros(shape, dtype=inp.dtype))


    modelOutputs = model(modelInputs)

    if (isinstance(modelOutputs, list)):
        logit = modelOutputs[0]
    else:
        logit = modelOutputs

    greedypred = tf.argmax( logit,axis=-1 )

    seq_length = tf.zeros((bs,),dtype=tf.int32) + tf.shape(logpower)[1]

    loss = tf.reduce_mean( tf.nn.ctc_loss( labels=sparsetarget,inputs=logit,sequence_length=seq_length, ctc_merge_repeated=True,time_major=False) )

    #global_step = tf.Variable(0, trainable=False)
    global_step = tf.train.get_or_create_global_step()
    starter_learning_rate = 1e-4 * hvd.size() * (bs/32.0)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.99, staircase=False)
    #learning_rate = tf.Print(learning_rate, [learning_rate], "learning_rate : ")

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    opt = hvd.DistributedOptimizer(opt)
    train_op = opt.minimize(loss, global_step=global_step)


    bcast = hvd.broadcast_global_variables(0)

    with tf.Session( config=config) as sess:
        nbepoch = 1000
        sess.run(noise_itr.initializer)
        sess.run(tf.global_variables_initializer())
        sess.run( bcast )
        #while not mon_sess.should_stop():
        # Perform synchronous training.
        for i in range(nbepoch):
            co = 0
            sess.run(itr.initializer)
            while True:
                try:
                    if (co % 10 == 0 and hvd.rank() == 0):
                        print("epoch : " + str(i))
                        print("batch " + str(co))

                    _, _loss, _gp, _s, _lr = sess.run([train_op, loss, greedypred, s,learning_rate])
                    if ( co % 10 == 0 and hvd.rank() == 0):
                        print("learning rate :" + str(_lr))
                        print("loss : " + str(_loss))
                        print("target : ")
                        # print( _s[0] )
                        print("".join([chr(x) for x in _s[0] if x != vocSize - 1]))
                        # print("greedy : ")
                        # print( _gp[0] )
                        ndgp0 = removeDoublons(_gp[0])
                        print("gp[0]: ")
                        print( "".join([chr(x) for x in ndgp0 if x != vocSize - 1]))

                    co = co + 1

                except tf.errors.OutOfRangeError:
                    break
            if (i % 1 == 0 and hvd.rank() == 0):
                print("model saved to disk")
                # saver.save(sess, "datasets/model2.ckpt")
                modelstringname = "commonvoice_fr_train"
                model.save_weights('datasets/pw-' + modelstringname + '.weights')

    return None





def demo():
    buildDatasetCommonVoiceIfNotExist("fr","train", 160000, 622)
    learn()


    return None

if __name__=="__main__":
    demo()