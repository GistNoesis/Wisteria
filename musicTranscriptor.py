import tensorflow as tf
import os.path
import numpy as np
import time

import errno
from mido import MidiFile,MidiTrack,Message,MetaMessage
import mido
from matplotlib import pyplot as plt
from subprocess import call
import scipy
import scipy.io
import scipy.io.wavfile
import sounddevice as sd
import sys
from tensorflow import keras as K
import glob
import argparse
import json

def stft(x, fftsize=1024, overlap=4):
    hop = int(fftsize / overlap)
    w = scipy.hanning(fftsize+1)[:-1]     # better reconstruction with this trick +1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize+1, hop)])

def istft(X, overlap=4):
    fftsize=(X.shape[1]-1)*2
    hop = int(fftsize / overlap)
    print(hop)
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop)
    for n,i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


def spectogram( x,fftsize = 1024, overlap = 4 ):
    w = tf.constant( value= scipy.hanning(fftsize+1)[:-1], dtype=tf.float32 )
    hop = fftsize // overlap
    shape1 =tf.shape(x)[1]
    dim1 = tf.cast( (shape1-fftsize) /hop,tf.int32) + 1
    grid = tf.reshape(tf.range(0,fftsize, dtype=tf.int32), (1, 1, fftsize))
    grid1 = tf.reshape( tf.range(0,dim1,dtype=tf.int32)*hop,(1,dim1,1))
    grid = grid + grid1
    bgrid = tf.reshape(tf.range(0,tf.shape(x)[0], dtype=tf.int32), (tf.shape(x)[0], 1, 1)) * tf.shape(x)[1]
    grid = bgrid+grid
    flatdata = tf.reshape(x,(-1,))
    flatgrid = tf.reshape(grid,(-1,))

    out = tf.gather(flatdata,flatgrid)
    rshp = tf.reshape( out, (tf.shape(x)[0],dim1,fftsize))
    rshpinpfft = tf.reshape( rshp,(tf.shape(x)[0]*dim1,fftsize))
    rshpw = tf.expand_dims(w,0)
    rfft = tf.spectral.rfft( rshpinpfft*rshpw)
    specter = tf.reshape( rfft,(x.shape[0],dim1,-1))

    return specter


def eventSequenceToPianoRoll( ticks_per_beat,evts):
    out = []
    cur = np.zeros((128,),dtype=np.uint8)
    curtime = 0
    curtempo = 500000
    eps = 1e-9

    # fftSize / (SampleRate* overlap)
    spectime = 1024.0 / (16000.0 * 4.0 )

    absolutemiditime = 0

    atleastonenote = False

    for evt in evts:
        #print(evt)
        absolutemiditime = absolutemiditime + mido.tick2second( evt.time , ticks_per_beat=ticks_per_beat,tempo=curtempo )
        #print(absolutemiditime)
        while curtime + eps < absolutemiditime  :
            out.append( cur )
            cur = np.copy(cur)
            curtime += spectime

        if (evt.type == "set_tempo"):
            curtempo = evt.tempo
        elif( evt.type == "note_on"):
            atleastonenote = True
            cur[ evt.note ] = evt.velocity
        elif(evt.type == "note_off"):
            cur[evt.note] = 0

    #print( absolutemiditime )

    if( len( out ) > 0 and atleastonenote):
        return np.stack(out,0)
    return None


def generateRandomMidi( filename, dsconfig ):
    nbbeats = 10
    #notes = np.arange(nbnotes) + 80
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 32
    #74 is flute
    #0 is piano

    instrument = np.random.choice( dsconfig["instruments"] )
    track.append(Message('program_change', program=instrument, time=0))
    track.append(MetaMessage('set_tempo', tempo=1000000, time=0))

    maxduration = nbbeats*mid.ticks_per_beat
    endtime = 0
    while endtime < maxduration:
        note= np.random.randint(40,100)
        velocity = np.random.randint(70,127)
        curendtime = endtime
        duration = np.random.randint(6,50)
        endtime = endtime+duration
        endtime = min(endtime,maxduration)
        duration = endtime-curendtime
        #we sometimes add silences
        if( np.random.rand() > 0.9):
            velocity = 0
        track.append(Message('note_on', note=note, velocity=velocity, time=0))
        track.append(Message('note_off', note=note, velocity=127, time=duration))

    pianoroll = eventSequenceToPianoRoll(mid.ticks_per_beat,track)

    mid.save(filename)
    return pianoroll

def generateRandomMidiSuperposition( filename,dsconfig ):
    nbbeats = 10
    #notes = np.arange(nbnotes) + 80
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 32
    #74 is flute
    #0 is piano
    instrument = np.random.choice( dsconfig["instruments"] )
    track.append(Message('program_change', program=instrument, time=0))
    track.append(MetaMessage('set_tempo', tempo=1000000, time=0))

    maxduration = nbbeats*mid.ticks_per_beat
    endtime = 0

    bpr = np.zeros((maxduration,128))

    nbnotes = np.random.randint(5,50)
    for i in range(nbnotes):
        note = np.random.randint(40,100)
        start = np.random.randint(nbbeats*mid.ticks_per_beat)
        duration = np.random.randint(6,50)
        bpr[ start:(start+duration),note ] = 1

    cur = np.zeros((128,),dtype=np.int32)

    dt = 0
    for i in range(maxduration):
        for j in range(128):
            if bpr[i,j]-cur[j] > 0:
                velocity = np.random.randint(70, 128)
                track.append(Message('note_on', note=j, velocity=velocity, time=dt))
                dt = 0
            elif  bpr[i,j]-cur[j] < 0:
                track.append(Message('note_off', note=j, velocity=127, time=dt))
                dt=0
        dt +=1
        cur = bpr[i]

    for j in range(128):
        if( cur[j] > 0):
            track.append(Message('note_off', note=j, velocity=127, time=dt))
            dt = 0
    track.append(Message('note_off', note=0, velocity=127, time=dt))

    pianoroll = eventSequenceToPianoRoll(mid.ticks_per_beat,track)

    #plt.imshow(np.transpose( pianoroll, (1, 0)),aspect=10)
    #plt.show()
    mid.save(filename)
    return pianoroll


def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_npy_file(item):
    data = np.load(item.decode())
    return data.astype(np.float32)

availableNames = ["randomMonophonic","randomPolyphonic"]


def getStringName( name,type,nbex):
    stringname = name + "_" + type + "_" + str(nbex)
    return stringname

def getDatasetNameFromConfig( dsconfig ):
    suffix = "" if "outputNameSuffix" not in dsconfig else dsconfig["outputNameSuffix"]
    return dsconfig["name"] + "_" + dsconfig["type"] +"_" + str( dsconfig["nbex"]) + "_" + suffix

def getModelNameFromConfig( modconfig ):
    suffix = "" if "outputNameSuffix" not in modconfig else modconfig["outputNameSuffix"]
    return modconfig["architecture"] + "_" + str(modconfig["depth"]) +"_" + str( modconfig["layerSize"]) + "_" + suffix

def getTrainingNameFromConfig( trconfig ):
    return trconfig["outputName"]


def getModelStringName( dsconfig,modconfig, trconfig ):
    dsstringname = getDatasetNameFromConfig(dsconfig)
    trainingstringname = getTrainingNameFromConfig(trconfig)
    modelstringname = dsstringname + "-" + getModelNameFromConfig(modconfig) + "-" + trainingstringname
    return modelstringname


def buildDatasetIfNotExist( dsconfig):
    if( dsconfig["name"] not in availableNames):
        raise Exception("name must be in availableNames")
    name = dsconfig["name"]

    stringname = getDatasetNameFromConfig(dsconfig)
    filename = "datasets/" + stringname + ".tfrecords"
    if( os.path.isfile(filename) ):
        return
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(dsconfig["nbex"]):
            #if( index % 1000 == 0):
            print("generating example " + str(index))
            dir = stringname + "/"+str(int(index/1000000)) + "/" + str(int(index/1000)) + "/"
            make_dir("datasets/" + dir)

            midifilename = dir + "pianoroll"+str(index)+".mid"
            if name == "randomMonophonic":
                pianoroll = generateRandomMidi("datasets/"+midifilename,dsconfig)
            elif name == "randomPolyphonic":
                pianoroll = generateRandomMidiSuperposition("datasets/" + midifilename,dsconfig)

            print( pianoroll.shape )
            wavfilepath =  dir + "audio"+str(index)+".wav"
            call(["fluidsynth", "-r", "16000", "-F", "datasets/"+wavfilepath, "/usr/share/sounds/sf2/FluidR3_GM.sf2", "datasets/"+midifilename])

            path = dir + "pianoroll" + str(index) + ".npy"
            np.save("datasets/" + path,pianoroll)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'pianoroll_path': _bytes_feature(tf.compat.as_bytes(path)),
                        'midifile_path': _bytes_feature(tf.compat.as_bytes(midifilename)),
                        'wavfile_path': _bytes_feature(tf.compat.as_bytes(wavfilepath))
                    }))
            writer.write(example.SerializeToString())

    return


def buildNoiseDatasetIfNotExist():
    filename = "datasets/noise.tfrecords"
    if( os.path.isfile(filename) ):
        return
    with tf.python_io.TFRecordWriter(filename) as writer:
        for file in glob.glob("datasets/noise/rnnoise_contributions/*.raw"):
            #ffmpeg -f s16le -ar 44.1k -ac 2 -i 1531117295146-office.raw -ar 16000 out.wav
            wavname = file[:-3]+"wav"
            call(["ffmpeg","-f","s16le","-ar","44.1k","-ac","2","-i",file,"-ar","16000",wavname])
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'wavfile_path': _bytes_feature(tf.compat.as_bytes(wavname))
                    }))
            writer.write(example.SerializeToString())
    return


def read_wav_file(filename):
    rate,data = scipy.io.wavfile.read(filename, mmap=False)
    #print(rate)
    #print("datashape ")
    #print(data.shape)
    #print(data.dtype)
    return data.astype(np.float32) / 32768.0



def make_iterator(bs,stringname):
    filename = "datasets/"+stringname+".tfrecords"
    dataset = tf.data.TFRecordDataset([filename])

    def record_parser(record):
        features = {"pianoroll_path": tf.FixedLenFeature((), tf.string, default_value=""),
                    "midifile_path": tf.FixedLenFeature((), tf.string, default_value=""),
                    "wavfile_path": tf.FixedLenFeature((), tf.string, default_value=""),
                    }
        parsed_features = tf.parse_single_example(record, features)
        pianoroll_path = parsed_features["pianoroll_path"]
        midifile_path = parsed_features["midifile_path"]
        wavfile_path = parsed_features["wavfile_path"]
        path = tf.strings.join(["datasets/", pianoroll_path])
        pianoroll = tf.py_func(read_npy_file, [path], [tf.float32,])
        pianoroll = tf.reshape( pianoroll,(-1,128)) / 127.0

        pathwav = tf.strings.join(["datasets/", wavfile_path])
        wavfile = tf.py_func(read_wav_file,[pathwav],[tf.float32])
        wavfile = tf.squeeze(wavfile,axis=0)
        #wavfile = tf.Print(wavfile,[tf.shape(wavfile)],"tfshapewavfile",summarize=5)
        #img = tf.io.decode_raw( parsed_features["image_data"],tf.float32 )

        #img = tf.io.decode_raw( )
        return midifile_path,pianoroll,wavfile

    dataset = dataset.shuffle(buffer_size=1000000)
    dataset = dataset.map(record_parser,num_parallel_calls=10)
    dataset = dataset.batch(bs,drop_remainder=True)
    dataset = dataset.prefetch(10*bs)
    #dataset = dataset.repeat(1)
    iterator = dataset.make_initializable_iterator()
    return iterator

def make_noise_iterator(bs,length):
    filename = "datasets/noise.tfrecords"
    dataset = tf.data.TFRecordDataset([filename])

    def record_parser(record):
        features = {
                    "wavfile_path": tf.FixedLenFeature((), tf.string, default_value=""),
                    }
        parsed_features = tf.parse_single_example(record, features)
        wavfile_path = parsed_features["wavfile_path"]

        wavfile = tf.py_func(read_wav_file,[wavfile_path],[tf.float32])
        wavfile = tf.squeeze(wavfile,axis=0)
        wavfile = tf.concat([wavfile[:,0],tf.zeros((160000,))],axis=0)
        #out = tf.zeros( (length,))
        out = wavfile[0:length]

        #wavfile = tf.Print(wavfile,[tf.shape(wavfile)],"tfshapewavfile",summarize=5)
        #img = tf.io.decode_raw( parsed_features["image_data"],tf.float32 )

        #img = tf.io.decode_raw( )
        return out

    dataset = dataset.shuffle(buffer_size=1000000)
    dataset = dataset.map(record_parser,num_parallel_calls=10)
    dataset = dataset.batch(bs,drop_remainder=True)
    dataset = dataset.prefetch(10*bs)
    dataset = dataset.repeat(-1)
    iterator = dataset.make_initializable_iterator()
    return iterator

def resBlockConv1DRelu(x, layerSize, kernelSize,dilation, istraining):
    act = tf.nn.relu
    #h = act( tf.layers.BatchNormalization()( tf.layers.Conv1D( layerSize,kernelSize,dilation_rate=dilation, padding="SAME", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN') )(x),training=istraining ))
    #h = x + act( tf.layers.BatchNormalization()( tf.layers.Conv1D(layerSize, kernelSize,dilation_rate=dilation, padding="SAME", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN') )(h),training=istraining) )
    h = act(tf.contrib.layers.instance_norm(
        tf.layers.Conv1D(layerSize, kernelSize, dilation_rate=dilation, padding="SAME")(
            x)))
    h = x + act(tf.contrib.layers.instance_norm(
        tf.layers.Conv1D(layerSize, kernelSize, dilation_rate=dilation, padding="SAME")(
            h)))

    return h

def resBlockConv1DSelu(x, layerSize, kernelSize,dilation, istraining):
    act = tf.nn.selu
    #h = act( tf.layers.BatchNormalization()( tf.layers.Conv1D( layerSize,kernelSize,dilation_rate=dilation, padding="SAME", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN') )(x),training=istraining ))
    #h = x + act( tf.layers.BatchNormalization()( tf.layers.Conv1D(layerSize, kernelSize,dilation_rate=dilation, padding="SAME", kernel_initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN') )(h),training=istraining) )
    h = act(
        tf.layers.Conv1D(layerSize, kernelSize, dilation_rate=dilation, padding="SAME",
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN'))(
            x))
    h = x + act(
        tf.layers.Conv1D(layerSize, kernelSize, dilation_rate=dilation, padding="SAME",
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN'))(
            h))

    return h


def resBlockKerasConv1DSelu( x, layerSize,kernelSize,dilation):
    h = tf.keras.layers.Conv1D(layerSize, kernelSize, activation=tf.keras.activations.selu,padding="SAME", dilation_rate=dilation,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',distribution="normal"))(x)
    h = tf.keras.layers.Conv1D(layerSize, kernelSize, activation=tf.keras.activations.selu,padding="SAME",dilation_rate=dilation,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',distribution="normal"))(h)
    h = tf.keras.layers.Add()([x,h])
    #h = tf.keras.layers.Activation(tf.keras.activations.selu)(h)
    return h


def preprocesslp(logpower):
    #minlogpower = tf.reduce_min(tf.reduce_min(logpower, axis=1, keep_dims=True), axis=2, keep_dims=True)
    #maxlogpower = tf.reduce_max(tf.reduce_max(logpower, axis=1, keep_dims=True), axis=2, keep_dims=True)
    #logpower = (logpower - minlogpower) / (1.0 + maxlogpower - minlogpower)
    return logpower




def QTransform( x, freq):
    shp = tf.shape(x)
    bs = shp[0]
    tdim = shp[1]
    featdim = shp[2]
    freqdim = tf.shape(freq)[0]
    flatx = tf.reshape(x,(-1,))

    rbs = tf.reshape( tf.range(bs,dtype=tf.int32), (bs,1,1) )
    rtdim = tf.reshape( tf.range(tdim,dtype=tf.int32), (1,tdim,1) )
    lowfreq =  tf.reshape( tf.math.floor(freq),(1,1,-1))
    highfreq = tf.reshape( tf.math.ceil(freq),(1,1,-1))
    w = freq - lowfreq
    lowind = tf.reshape( rbs*tdim*featdim + rtdim*featdim + tf.cast(lowfreq,dtype=tf.int32),(-1,))
    highind = tf.reshape( rbs*tdim*featdim + rtdim*featdim + tf.cast(highfreq,dtype=tf.int32),(-1,))

    lowgather =tf.reshape( tf.gather( flatx,lowind ), (bs,tdim,freqdim))
    highgather = tf.reshape( tf.gather( flatx,highind),(bs,tdim,freqdim))
    out = w*highgather + (1-w)*lowgather
    return out


class QTransformLayer(tf.keras.layers.Layer):
    def __init__(self,nbfeat):
        super(QTransformLayer, self).__init__()
        self.nbfeat = nbfeat


    def build(self, input_shape):
        self.built=True
        return

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.nbfeat)

    def call(self, input):
        halftone = tf.constant( np.power(2,(1/12)),tf.float32)
        # f[k} =k/(NT)
        # f[k]= k / (1024*(1/16000))
        # note 69 = A4 = 440hz => k = 440hz * (1024/ 16000) = 28.16

        f0 = tf.constant( (440.0 / 16000.0) * 1024.0,tf.float32)
        freq = f0 * tf.pow( halftone, tf.range(self.nbfeat, dtype=tf.float32)-69 )
        out = QTransform(input,freq)
        return out







def buildModel( modconfig ):
    if( modconfig["architecture"] == "conv1"):
        return modelConv1( modconfig["depth"],modconfig["layerSize"])
    if( modconfig["architecture"] == "GRU"):
        return modelGRU( modconfig["depth"],modconfig["layerSize"] )
    raise Exception("architecture not recognized")


def GruCollect( x, size, initialStates,outStates):
    initialState = tf.keras.Input(shape=( size,), dtype=tf.float32, name='GRUState'+str(len(initialStates)))
    out,outState = tf.keras.layers.GRU(size,return_sequences=True,return_state=True)( x , initial_state= [initialState])
    initialStates.append( initialState )
    outStates.append(outState)
    return out

def modelGRU( depth,layerSize):
    outSize = 128
    initialStates = []
    outStates = []
    kernelSize = 1
    logspecter = tf.keras.Input(shape=(None,513), dtype=tf.float32, name='logspectrum')
    h = tf.keras.layers.Conv1D(layerSize, kernelSize, padding="SAME")(logspecter)
    for i in range(depth):
        h = GruCollect(h,layerSize,initialStates,outStates)

    out = tf.keras.layers.Conv1D(outSize, kernelSize, padding="SAME")(h)

    model = tf.keras.Model(inputs=[logspecter] + initialStates , outputs=[out]+outStates)
    return model


def modelConv1( depth,layerSize):
    kernelSize = 1
    outSize = 128

    logspecter = tf.keras.Input(shape=(None, 513), dtype=tf.float32, name='logspectrum')

    nbnotefreq = 128

    # logspecter = QTransform( logspecter, freq)
    conv2d = False
    if (conv2d):
        qtransfo = QTransformLayer(nbnotefreq)(logspecter)
        feat2dDim = 3
        rshp2d = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 3), lambda shp: (shp[0], shp[1], shp[2], 1))(
            qtransfo)
        h2d = tf.keras.layers.Conv2D(layerSize, (1, 40), activation=tf.keras.activations.selu, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                                              distribution="normal"))(
            rshp2d)
        h2d = tf.keras.layers.Conv2D(feat2dDim, (1, 40), activation=tf.keras.activations.selu, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                                              distribution="normal"))(
            h2d)
        h = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], feat2dDim * nbnotefreq)),
                                   lambda shp: (shp[0], shp[1], shp[2] * shp[3]))(h2d)

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


def learn( dsconfig, modconfig, trconfig):
    dsstringname = getDatasetNameFromConfig(dsconfig)
    modelstringname = getModelStringName(dsconfig,modconfig,trconfig)

    g1 = tf.Graph()
    with g1.as_default():
        bs = 32
        itr = make_iterator(bs,dsstringname)
        data = itr.get_next()



        lr = trconfig["initialLearningRate"]

        X = tf.placeholder(shape=(bs, None), dtype=tf.float32)
        target = tf.placeholder(shape=(bs, None,128), dtype=tf.float32)
        istraining = tf.placeholder(dtype=tf.bool,shape=())

        if( trconfig["noise"] is True):
            noise_itr = make_noise_iterator(bs, 160000)
            noisedata = noise_itr.get_next()
            X = X + tf.random.uniform((bs,1),0.0,4.0) * noisedata

        if( trconfig["volumeRandomization"] is True):
            volgain = 0.1+tf.random.uniform((bs,1),0,2)
            X = X*volgain

        if( trconfig["whiteNoise"] is True):
            varnoise = 0.1*tf.random.normal( tf.shape(X) )
            noise = tf.random.normal( tf.shape(X) )*varnoise*varnoise
            X = X + noise


        fft = spectogram(X)
        power = tf.real( fft*tf.math.conj(fft) )
        power = tf.reshape(power,(bs,-1,513))

        if (trconfig["distortion"] is True):
            distort = 1 + tf.random.normal( tf.shape(power), )*0.05
            power = distort * power

        #cutofffreq = tf.random.uniform((bs,1,1),minval=100,maxval=400)

        #micFrequencyResponse = tf.random.uniform((bs,1,1),-10,0) + tf.cumsum( tf.random.normal((bs,1,513))*0.05, axis=1 )



        #attenuate = 1 / (1 + tf.pow(tf.reshape(tf.range(0,513,dtype=tf.float32), (1, 1, 513))/cutofffreq, tf.random.uniform((bs,1,1 ),0.0, 2.0) )  )
        #power = attenuate * power

        #power = tf.Print( power ,[tf.reduce_mean( power )],"power :" )
        #pinknoise =  tf.random.normal( (bs,tf.shape(power)[1],513) ) *  1.0 /( 1+tf.reshape( tf.range(0,513,dtype=tf.float32),(1,1,513) ) )
        #power = power + tf.random.uniform((bs,1,1), 0,1e-8) * pinknoise

        eps = 1e-10
        logpower = tf.log(eps+power)

        if trconfig["micFrequencyResponse"] is True:
            nbcp = 10
            micFRCP = tf.random.uniform((bs, 1, nbcp), -10.0, 1)
            micFrequencyResponse = QTransform(micFRCP, (nbcp - 1.01) * tf.range(513, dtype=tf.float32) / 512.0)
            logpower = logpower + micFrequencyResponse


        logpower = preprocesslp(logpower)

        model = buildModel( modconfig )

        modelInputs = [logpower]

        for i in range(len(model.inputs)-1):
            inp = model.inputs[i+1]
            shape = (bs,) + tuple( inp.shape[1:] )
            modelInputs.append( tf.zeros( shape, dtype=inp.dtype))

        modelOutputs = model(modelInputs)

        if( isinstance(modelOutputs,list)):
            logit = modelOutputs[0]
        else:
            logit = modelOutputs

        tdim = tf.shape(logit)[1]
        targ = target[:, 0:tdim, :]

        eps = 1e-5
        #loss = tf.losses.mean_squared_error(targ,pred)
        loss =  tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.greater_equal(targ,eps), tf.float32),logits=logit ))

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   1000, trconfig["decayLearningRate1000"], staircase=False)
        learning_rate = tf.Print(learning_rate,[learning_rate],"learning_rate : ")
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

        #saver = tf.train.Saver(tf.global_variables(scope="model"))


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            start_time = time.time()
            # your code

            nbepoch = trconfig["nbEpoch"]

            for i in range(nbepoch):
                co = 0
                sess.run(itr.initializer)
                while True:
                    try:
                        print("epoch : " + str(i))
                        print("batch " + str(co))
                        m,p,w = sess.run(data)

                        #sp = sess.run( fft,feed_dict={X:w[:,:,0]} )
                        _,_loss = sess.run( [train_op,loss],feed_dict={X:w[:,:,0], target:p,istraining:True} )
                        print("loss : " + str(_loss) )
                        #print( sp.shape)
                        #print( pianoroll.shape)
                        #plt.imshow(pianoroll[0])
                        #plt.show()
                        #print(name)
                        #print( np.sum(img))
                        co = co+1

                    except tf.errors.OutOfRangeError:
                        break
                if( i % 1 == 0):
                    print("model saved to disk")
                    #saver.save(sess, "datasets/model2.ckpt")
                    model.save_weights('datasets/pw-'+modelstringname+'.weights')

            print("model saved to disk")
            #saver.save(sess, "datasets/model2.ckpt")
            model.save_weights('datasets/fw-'+ modelstringname +'.weights')
        elapsed_time = time.time() - start_time
        print("Training time : ")
        print( elapsed_time )


def validate(name, type, nbex):
    stringname = getStringName(name, type, nbex)
    g3 = tf.Graph()
    with g3.as_default():
        bs = 1
        itr = make_iterator(bs,stringname)
        data = itr.get_next()

        lr = 1e-4

        X = tf.placeholder(shape=(bs, None), dtype=tf.float32)
        target = tf.placeholder(shape=(bs, None, 128), dtype=tf.float32)
        istraining = tf.placeholder(dtype=tf.bool, shape=())

        fft = spectogram(X)
        power = tf.real(fft * tf.math.conj(fft))

        power = tf.reshape(power, (bs, -1, 513))
        eps = 1e-10
        logpower = tf.log(eps + power)

        logpower = preprocesslp(logpower)


        model = buildModel()
        logit = model(logpower)

        pred = tf.nn.sigmoid(logit)
        tdim = tf.shape(logit)[1]
        targ = target[:, 0:tdim, :]

        loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.greater_equal(targ, eps), tf.float32),
                                                           logits=logit) )

        #saver = tf.train.Saver(tf.global_variables(scope="model"))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, "datasets/model2.ckpt")
            model.load_weights("datasets/model.weights")
            start_time = time.time()
            # your code

            nbepoch = 1

            for i in range(nbepoch):
                co = 0
                sess.run(itr.initializer)
                while True:
                    try:
                        print("epoch : " + str(i))
                        print("batch " + str(co))
                        m, p, w = sess.run(data)

                        # sp = sess.run( fft,feed_dict={X:w[:,:,0]} )
                        _loss,_pred = sess.run([ loss,pred], feed_dict={X: w[:, :, 0], target: p, istraining: True})
                        print("loss : " + str(_loss))
                        # print( sp.shape)
                        # print( pianoroll.shape)
                        plt.figure(0)
                        plt.imshow(np.transpose( (p[0] > 1e-5).astype(np.float32) ,(1,0)) ,origin='lower', vmin=0.0,vmax=1.0, aspect=10)

                        plt.figure(1)
                        plt.imshow(np.transpose( _pred[0], (1, 0)), origin='lower', vmin=0.0, vmax=1.0,aspect=10)
                        plt.show()
                        # print(name)
                        # print( np.sum(img))
                        co = co + 1

                    except tf.errors.OutOfRangeError:
                        break

        elapsed_time = time.time() - start_time
        print("Training time : ")
        print(elapsed_time)




def demorecord():
    volumegain = 1.0
    g2 = tf.Graph()
    with g2.as_default():
        with  tf.Session() as sess:
            K.backend.set_session(sess)
            #we choose 16khz it is ~ 16*1024 = 16384
            fs = 16000
            duration = 1.0# seconds
            nbspecter = int( 20.0/duration )

            bs = 1
            X = tf.placeholder(shape=(bs, None), dtype=tf.float32)
            istraining = tf.placeholder(dtype=tf.bool, shape=())

            fft = spectogram(X)
            power = tf.real( fft*tf.math.conj(fft) )
            power = tf.reshape(power,(bs,-1,513))
            eps = 1e-10
            logpower = tf.log(eps+power)

            logpower = preprocesslp(logpower)

            model = buildModel()

            logit = model( logpower )
            pred = tf.nn.sigmoid(logit)

            plt.ion()

            lastSpecter = []
            pianorolls = []
            #saver = tf.train.Saver(tf.global_variables(scope="model"))


            sess.run(tf.global_variables_initializer())
            model.load_weights("datasets/model.weights")
            #saver.restore(sess, "datasets/model2.ckpt")
            lp, pr = sess.run( [logpower,pred], feed_dict={X:np.random.randn(1,int(fs*duration)),istraining:False})
            lp2, pr2 = sess.run( [logpower,pred], feed_dict={X: np.random.randn(1, 2*int(fs * duration)),istraining:False})

            dim1dur = lp[0].shape[0]
            dim2dur = lp2[0].shape[0]

            dimoverlap = dim2dur-dim1dur

            dimoverlappr = pr2[0].shape[0] - pr[0].shape[0]

            sp = np.transpose(lp[0], (1, 0))
            sp2 = np.transpose(lp2[0][0:dimoverlap,:], (1, 0))

            pr = np.transpose( pr[0],(1,0) )
            pr2 = np.transpose(pr2[0][0:dimoverlappr,:],(1,0))

            for i in range(nbspecter-1):
                lastSpecter.append(sp2)
                pianorolls.append( np.random.uniform(0,1,pr2.shape ) )

            lastSpecter.append(sp)
            pianorolls.append(pr)


            plt.figure(0)
            im = plt.imshow( np.concatenate( lastSpecter[-nbspecter:],axis=1) ,origin='lower')
            plt.figure(1)
            impr = plt.imshow( np.concatenate( pianorolls[-nbspecter:],axis=1) ,origin='lower', vmin=0.0,vmax=1.0, aspect=10)


            plt.show(block=False)

            plt.pause(0.01)

            allaudio = []


            for i in range(10000):
                myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2,blocking = True)
                print( "max myrecording ")
                print( np.max(myrecording))
                allaudio.append(myrecording)
                lp,pr = sess.run([logpower,pred], feed_dict={X: volumegain*np.expand_dims(myrecording[:,0],axis=0),istraining:False} )
                lp2,pr2 = sess.run([logpower,pred], feed_dict={X: volumegain*np.expand_dims(np.concatenate(allaudio[-2:],axis=0)[:,0] , axis=0),istraining:False})

                print("max pred :")
                print( np.max(pr))
                print("min pred : ")
                print( np.min(pr))

                lastSpecter[-1] = np.transpose(lp2[0][0:dimoverlap,:],(1,0))
                lastSpecter.append( np.transpose(lp[0], (1, 0)))

                pianorolls[-1] = np.transpose(pr2[0][0:dimoverlappr,:],(1,0))
                pianorolls.append( np.transpose( pr[0],(1,0) ) )

                im.set_data(np.concatenate( lastSpecter[-nbspecter:] ,axis=1) )
                #impr.set_data( (np.concatenate( pianorolls[-nbspecter:],axis=1) > 0.8).astype(np.float32)  )
                impr.set_data(np.concatenate( pianorolls[-nbspecter:],axis=1)  )

                plt.show(block=False)
                plt.pause(0.01)

                print(myrecording.shape)

            #sound = np.concatenate(allaudio,axis=0)
            #sd.play( sound ,fs, blocking=True)


def exportModel( dsconfig, modconfig,trconfig ):
    modelstringname = getModelStringName(dsconfig,modconfig,trconfig)
    g2 = tf.Graph()
    with g2.as_default():
        with  tf.Session() as sess:
            model = buildModel(modconfig)
            model.load_weights("datasets/fw-"+modelstringname+".weights")
            import tensorflowjs as tfjs
            path = "tfjs/models/"+modelstringname
            make_dir(path)
            tfjs.converters.save_keras_model(model,path)

def testIterateNoise():
    bs = 32
    itr = make_noise_iterator(bs,160000)
    data = itr.get_next()
    sess = tf.Session()
    sess.run(itr.initializer)

    for i in range(10):
        noisywav = sess.run(data)
        print(np.shape(noisywav))


def loadJsonFile( path ):
    with open( path) as f:
        data = f.read()
        return json.loads(data)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", help="dataset config file")
    parser.add_argument("-mod", help="model config file")
    parser.add_argument("-tr", help="training config file")
    parser.add_argument("-act", help="l to learn, v to validate, e to export")
    args = parser.parse_args()

    dsconfig = None
    modconfig = None
    trconfig = None

    if( args.ds is not None):
        dsconfig = loadJsonFile(args.ds)

    if (args.mod is not None):
        modconfig = loadJsonFile(args.mod)

    if (args.tr is not None):
        trconfig = loadJsonFile(args.tr)

    print("Dataset config :")
    print( dsconfig)
    print("Model config :")
    print( modconfig)
    print("Training config :")
    print( trconfig)

    if( args.act == "l"):
        buildDatasetIfNotExist( dsconfig)
        buildNoiseDatasetIfNotExist()
        learn( dsconfig,modconfig,trconfig)
    if( args.act == "e"):
        exportModel(dsconfig,modconfig,trconfig)

    '''
    if( args.act == "l"):
        buildDatasetIfNotExist(args.dsname,args.dstype,args.dsnbex)
        buildNoiseDatasetIfNotExist()
        learn(args.dsname,args.dstype,args.dsnbex)
    elif( args.act == "v"):
        buildDatasetIfNotExist(args.dsname, args.dstype, args.dsnbex)
        validate(args.dsname,args.dstype,args.dsnbex)
    elif( args.act == "e"):
        modelname = getStringName(args.dsname,args.dstype,args.dsnbex)
        exportModel(modelname)
    '''
    '''
    buildDatasetIfNotExist()
    buildNoiseDatasetIfNotExist()
    if( len(sys.argv) > 1 and sys.argv[1]=="l"):
        learn()
    elif (len(sys.argv) > 1 and sys.argv[1] == "n"):
        testIterateNoise()
    elif (len(sys.argv) > 1 and sys.argv[1] == "v"):
        validate()
    elif (len(sys.argv) ==2 and sys.argv[1] == "e"):
        print("usage : musicTranscriptor e output")
    elif (len(sys.argv) > 2 and sys.argv[1] == "e"):
        exportModel(sys.argv[2])
    elif (len(sys.argv) > 1 and sys.argv[1] == "g"):
        generateRandomMidiSuperposition("randomMidiSuperposition.mid")
    #else:
    #    demorecord()
    '''