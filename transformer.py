import tensorflow as tf
import numpy as np


def buildDatasetRepeatWordSequenceInReverseUnsupervised(size=60000,nbWords = 5,voc_size = 256, usedWords = 10, lout = 300):
    #goal is to learn to transform from character level, word list to word list in reverse order
    #hello beautiful world -> world beautiful hello
    outx = []
    outseqLength = []

    dicpath = "/usr/share/dict/words"
    with open(dicpath, 'r') as f:
        allWords = f.read().splitlines()

    if( usedWords > 0):
        subWords = []
        for i in range( usedWords ):
            subWords.append( allWords[ np.random.randint(0,len(allWords))] )
        allWords = subWords

    for i in range(size):
        tempx = np.ones((lout,),dtype=np.uint8)*(voc_size-1)
        words = np.random.randint(0,len(allWords),size=(nbWords))
        cox = 0
        for j in range(nbWords):
            for k in allWords[words[j]]:
                tempx[cox] = ord(k)
                cox+=1
            tempx[cox] = ord(' ')
            cox +=1

        tempx[cox] = ord('=')
        cox += 1
        tempx[cox] = ord(' ')
        cox += 1

        for j in range(nbWords):
            for k in allWords[words[ (nbWords -1) - j ]]:
                tempx[cox] = ord(k)
                cox+=1
            tempx[cox] = ord(' ')
            cox +=1

        outx.append(tempx)
        outseqLength.append(cox)

    return ( np.stack(outx,axis=0),np.stack(outseqLength,axis=0) )





class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(LayerNorm, self).__init__()
        self.axis= axis

    def build(self, input_shape):
        self.g = self.add_variable("g",
                                        shape=[int(input_shape[-1]) ] , initializer=tf.constant_initializer(1) )
        self.b = self.add_variable("b",
                                   shape=[int(input_shape[-1])] , initializer=tf.constant_initializer(0) )
        return input_shape

    def get_config(self):
        base_config = super(LayerNorm, self).get_config()
        base_config['axis'] = self.axis
        return base_config

    def call(self, x):
        epsilon = 1e-5
        u = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=self.axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * tf.reshape(self.g,(1,1,-1)) + tf.reshape( self.b,(1,1,-1))
        return x

def positionEncodding( x,nbpast, d= 100 ):
    bs = tf.shape(x)[0]
    tdim = tf.shape(x)[1]
    #nbpast = tf.reshape( nbpast[0,0],(1,))
    offsetpast = tf.reshape( nbpast, (bs,1,1))
    pos = tf.cast( offsetpast,dtype=tf.float32) + tf.reshape( tf.range(tf.cast(tdim,tf.float32),dtype=tf.float32),(1,tdim,1) )
    #r1 = tf.Print(r1,[r1],"r1",summarize=10000)
    featrange = tf.reshape( tf.math.pow( 10000.0, tf.range(d,dtype=tf.float32) / d),(1,1,d))

    cos = tf.cos( pos / featrange )
    sin = tf.sin( pos / featrange )
    #return tf.tile( tf.expand_dims( tf.concat( [cos,sin], axis=1),axis=0),(bs,1,1))
    return tf.concat( [cos,sin],axis=2)

class LayerPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, d):
        super(LayerPositionEmbedding, self).__init__()
        self.d = d

    #def build(self, input_shapes):
    #    input_shape = input_shapes[0]
    #    return (input_shape[0],input_shape[1],2*self.d)
    def get_config(self):
        base_config = super(LayerPositionEmbedding, self).get_config()
        base_config['d'] = self.d
        return base_config

    def call(self, x):
        return positionEncodding(x[0],x[1],self.d)




def attention_mask(nd, ns, localAttention,*, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    #m = tf.Print(m, [m[0:20, 0:20]], "m", summarize=1000)
    if( localAttention > 0):
        oldm = i < j - ns + nd + localAttention
        #oldm = tf.Print(oldm,[oldm[0:20,0:20]],"oldm",summarize=1000 )
        m = tf.logical_and( m, oldm)
        #m = tf.Print(m, [m[0:20, 0:20]], "finalm", summarize=1000)


    return tf.cast(m, dtype)

def mask_attn_weights(w,localAttention):
    # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    nd = tf.shape(w)[2]
    ns = tf.shape(w)[3]
    b = attention_mask(nd, ns,localAttention, dtype=w.dtype)
    #b = tf.Print(b,[b[0:10,0:10]],"attention mask",summarize=10000)

    b = tf.reshape(b, [1, 1, nd, ns])
    w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
    return w

def selfMaskedAttention( q, k ,v, localAttention ):
    #q shape : (bs,nbhead, tdim, keydim)
    #k shape : (bs,nbhead, tdim, keydim)
    #v shape : (bs,nbhead, tdim, valdim)

    #out shape ! (bs, tdim, nbhead*valdim)

    w = tf.matmul(q, k, transpose_b=True)
    dimv = tf.shape(v)[-1]
    w = w * tf.rsqrt(tf.cast(dimv, w.dtype))
    w = mask_attn_weights( w ,localAttention)
    #w = tf.Print(w, [tf.shape(w)],"shape w",summarize=10)
    sm = tf.nn.softmax( w )

    out = tf.matmul( sm, v )

    out = tf.transpose( out, (0,2,1,3))
    out = tf.reshape( out, (tf.shape(out)[0],tf.shape(out)[1],tf.shape(out)[2]*tf.shape(out)[3] ))
    return out

'''
def selfAttention( x , nbhead, kdim, vdim ):

    bs = tf.shape(x)[0]
    tdim = tf.shape(x)[1]

    q = tf.reshape( tf.keras.layers.Conv1D(kdim*nbhead,kernel_size=1,use_bias=False)(x) , (bs,tdim,kdim,nbhead) )
    k = tf.reshape( tf.keras.layers.Conv1D(kdim*nbhead,kernel_size=1,use_bias=False)(x) , (bs,tdim,kdim,nbhead) )
    v = tf.reshape( tf.keras.layers.Conv1D(vdim*nbhead,kernel_size=1,use_bias=False)(x) , (bs,tdim,vdim,nbhead) )

    q = tf.transpose( q, (0, 3, 1, 2))
    k = tf.transpose( k, (0, 3, 1, 2))
    v = tf.transpose( v, (0, 3, 1, 2))

    out = selfMaskedAttention(q,k,v)

    out = tf.reshape( out, (bs,tdim,nbhead*vdim))

    return out
'''


class LayerMaskedAttention(tf.keras.layers.Layer):
    def __init__(self, nbhead, kdim, vdim,localAttention):
        super(LayerMaskedAttention, self).__init__()
        self.nbhead = nbhead
        self.kdim = kdim
        self.vdim = vdim
        self.localAttention = localAttention

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], input_shape[1], self.vdim * self.nbhead)

    def call(self, inputs):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]
        out = selfMaskedAttention(q, k, v,self.localAttention)
        bs = tf.shape(q)[0]
        tdim = tf.shape(q)[2]
        out = tf.reshape(out, (bs, tdim, self.nbhead * self.vdim))
        return out

    def get_config(self):
        base_config = super(LayerMaskedAttention, self).get_config()
        base_config['nbhead'] = self.nbhead
        base_config['kdim'] = self.kdim
        base_config['vdim'] = self.vdim
        base_config['localAttention'] = self.localAttention
        return base_config

class LayerSplitHead(tf.keras.layers.Layer):
    def __init__(self, nbhead, outdim ):
        super(LayerSplitHead, self).__init__()
        self.nbhead = nbhead
        self.outdim = outdim

    def call(self, x):
        bs = tf.shape(x)[0]
        tdim = tf.shape(x)[1]
        q = tf.reshape(x,(bs, tdim, self.outdim, self.nbhead) )
        out = tf.transpose(q, (0, 3, 1, 2))
        return out

    def get_config(self):
        base_config = super(LayerSplitHead, self).get_config()
        base_config['nbhead'] = self.nbhead
        base_config['outdim'] = self.outdim
        return base_config


def selfAttentionBlock( x,nbhead,kdim, vdim,localAttention, pastk, pastv, presentKs, presentVs ):
    q = LayerSplitHead(nbhead, kdim) ( tf.keras.layers.Conv1D(kdim * nbhead, kernel_size=1, use_bias=False)(x))
    k = LayerSplitHead(nbhead, kdim) ( tf.keras.layers.Conv1D(kdim * nbhead, kernel_size=1, use_bias=False)(x))
    v = LayerSplitHead(nbhead, vdim) ( tf.keras.layers.Conv1D(vdim * nbhead, kernel_size=1, use_bias=False)(x))

    presentKs.append(k)
    presentVs.append(v)

    k = tf.keras.layers.Reshape( (nbhead,-1,kdim))(k)
    v = tf.keras.layers.Reshape( (nbhead,-1,vdim))(v)

    k = tf.keras.layers.Concatenate(axis=2)([pastk,k])
    v = tf.keras.layers.Concatenate(axis=2)([pastv,v])



    out = LayerMaskedAttention(nbhead,kdim,vdim,localAttention)([q,k,v])
    return out


def transformerBlock( x,localAttention, nbhead,kdim, vdim , pastKs,pastVs, presentKs, presentVs ):
    pastK = tf.keras.layers.Input( shape=(nbhead,None,kdim))
    pastV = tf.keras.layers.Input( shape=(nbhead,None,vdim))

    pastKs.append( pastK )
    pastVs.append( pastV )

    att = selfAttentionBlock(x,nbhead,kdim,vdim,localAttention,pastK,pastV,presentKs,presentVs)
    h = tf.keras.layers.Add()([x,att])
    h0 = LayerNorm()(h)
    h1 = tf.keras.layers.Conv1D(vdim*nbhead, 1, activation=tf.keras.activations.selu)(h0)
    h2 = tf.keras.layers.Conv1D(vdim * nbhead, 1, )(h1)
    h = tf.keras.layers.Add()( [h0 , h2 ] )
    h = LayerNorm()(h)
    return h

#This code is just use for debug purpose to toggle or not some layers
def transformerBlock2( x,localAttention, nbhead,kdim, vdim , pastKs,pastVs, presentKs, presentVs ):
    pastK = tf.keras.layers.Input( shape=(nbhead,None,kdim))
    pastV = tf.keras.layers.Input( shape=(nbhead,None,vdim))

    pastKs.append( pastK )
    pastVs.append( pastV )

    #att = selfAttentionBlock(x,nbhead,kdim,vdim,localAttention,pastK,pastV,presentKs,presentVs)

    h = tf.keras.layers.Add()([x,x])
    h0 = LayerNorm()(h)
    h1 = tf.keras.layers.Conv1D(vdim*nbhead, 1, activation=tf.keras.activations.selu)(h0)
    h2 = tf.keras.layers.Conv1D(vdim * nbhead, 1, )(h1)
    h = tf.keras.layers.Add()( [h0 , h2 ] )
    #h = LayerNorm()(h)
    return h



def buildModel( layerSize,depth, vocSize, embSize, nbhead,kdim,localAttention):
    x = tf.keras.layers.Input( shape=(None,), dtype=tf.int32 )
    emb = tf.keras.layers.Embedding(vocSize, embSize)(x)
    nbpast = tf.keras.layers.Input( shape=(1,), dtype=tf.int32)

    #add past length to position embedding
    pos = LayerPositionEmbedding(layerSize // 2)([emb,nbpast])
    h = tf.keras.layers.Add()([pos, emb])

    pastKs = []
    pastVs = []
    presentKs = []
    presentVs = []

    for i in range(depth):
        h = transformerBlock( h,localAttention,nbhead,kdim, int(layerSize/nbhead),pastKs,pastVs,presentKs,presentVs)

    logit = tf.keras.layers.Conv1D(vocSize, 1)(h)

    model = tf.keras.Model( inputs=[x,nbpast] + pastKs + pastVs, outputs=[logit] + presentKs + presentVs)
    return model



def demo():
    ds = buildDatasetRepeatWordSequenceInReverseUnsupervised()

    for i in range(10):
        string = "".join([ chr(x)for x in ds[0][i] if x != 255 ])
        print(string)

    lr = 1e-4
    bs = 32
    vocSize = 256
    embSize = 100
    layerSize = 100
    depth = 6
    localAttention = 0

    nbhead = 4
    kdim = 20
    vdim = layerSize // nbhead

    text = tf.placeholder(dtype=tf.int32,shape=(None,None) )
    shiftedText = tf.concat( [tf.ones((tf.shape(text)[0],1),dtype=tf.int32 ),  text],axis=1)
    target = tf.concat([text, tf.zeros((tf.shape(text)[0], 1), dtype=tf.int32)], axis=1)


    model = buildModel(layerSize,depth,vocSize,embSize,nbhead,kdim,localAttention)

    pks = []
    pvs = []

    for i in range((len( model.inputs)-2) // 2):
        pks.append( tf.zeros((bs,nbhead,0,kdim)) )
        pvs.append( tf.zeros((bs,nbhead,0,vdim)) )

    outputs = model( [shiftedText,tf.zeros((bs,1),dtype=tf.int32)] + pks + pvs )
    logit = outputs[0]

    greedyPred = tf.argmax(logit, axis=-1)


    pks = []
    pvs = []

    for i in range((len(model.inputs) - 1) // 2):
        pks.append(tf.placeholder(shape=(bs, nbhead, None, kdim) ,dtype=tf.float32) )
        pvs.append(tf.placeholder(shape=(bs, nbhead, None, vdim) ,dtype=tf.float32) )

    nbpast = tf.placeholder(shape=(bs,1),dtype=tf.int32)

    generator = model( [text,nbpast] + pks + pvs )
    flatlogit = tf.reshape( generator[0], (-1,vocSize))
    generatorPred = tf.random.multinomial(flatlogit, 1,output_dtype=tf.int32)
    #generatorPred = tf.reshape( tf.argmax(flatlogit,axis=1),(-1,1))


    weights = tf.ones_like(shiftedText,dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(logit,target,weights)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)


    dataset = tf.data.Dataset.from_tensor_slices(ds)
    dataset = dataset.shuffle(ds[0].shape[0])
    dataset = dataset.batch(bs, True)
    iter = dataset.make_initializable_iterator()
    nextelement = iter.get_next()

    nbepoch = 5

    sess = tf.Session()
    sess.run( tf.global_variables_initializer() )




    for i in range(nbepoch):
        sess.run(iter.initializer)
        co = 0
        while True:
            try:
                print("epoch " + str(i))
                print("batch : " + str(co))
                batch = sess.run(nextelement)
                # print(batch[2])
                _,lossValue,greedy,pks0 = sess.run([ train_op, loss, greedyPred,outputs[1] ],
                                                feed_dict={text: batch[0]})
                print("loss : " + str(lossValue))
                print("greedy[0]: " + "".join([chr(x) for x in greedy[0] if x != 255]))
                #print("pks0[0,0,0:4,0] :"  + str( pks0[0,0,0:4,0] ))

                # input()
            except tf.errors.OutOfRangeError:
                break
            co += 1


    seqLen = 300
    sess.run(iter.initializer)
    batch = sess.run(nextelement)

    _pks = [np.zeros((bs, nbhead, 0, kdim)) for k in pks]
    _pvs = [np.zeros((bs, nbhead, 0, vdim)) for v in pvs]
    curText = np.ones((bs, 1), dtype=np.int32)
    seqs = []
    for i in range(seqLen):
        feed_dict = {}
        for var, value in zip(pks, _pks):
            feed_dict[var] = value
        for var, value in zip(pvs, _pvs):
            feed_dict[var] = value
        #if( i ==0 or i >=10):
        feed_dict[text] = curText
        #else:
        #    feed_dict[text] = batch[0][:,(i-1):i]
        feed_dict[nbpast] = i*np.ones((bs,1),dtype=np.int32)
        results = sess.run([generatorPred] + generator, feed_dict=feed_dict)
        # print( results[0])
        seqs.append(results[0])
        curText = results[0]
        for j in range(len(_pks)):
            _pks[j] = np.concatenate([_pks[j], results[2 + j]], axis=2)
            # print( _pks[j].shape)
        for j in range(len(_pvs)):
            _pvs[j] = np.concatenate([_pvs[j], results[2 + len(pks) + j]], axis=2)
            # print(_pvs[j].shape)
        #input()

        #print("pks0[0,0,0:4,0] :" + str(_pks[0][0, 0, 0:4, 0]))

    samples = np.stack(seqs, axis=1)
    print(samples.shape)
    for i in range(samples.shape[0]):
        print("samples[" + str(i) + " ]: " + "".join([chr(x) for x in samples[i] if x != 255]))

    input()


    #input()


    #text = tf.keras.Input(shape=(None, 256), dtype=tf.float32, name='text')

    #m = buildModel()

    #res = m(text)




if __name__=="__main__":
    demo()
