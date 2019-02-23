import * as tf from '@tensorflow/tfjs';
/*
class LayerMaskedAttention(tf.keras.layers.Layer):
    def __init__(self, nbhead, kdim, vdim):
        super(LayerMaskedAttention, self).__init__()
        self.nbhead = nbhead
        self.kdim = kdim
        self.vdim = vdim

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], input_shape[1], self.vdim * self.nbhead)

    def call(self, inputs):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]
        out = selfMaskedAttention(q, k, v)
        bs = tf.shape(q)[0]
        tdim = tf.shape(q)[2]
        out = tf.reshape(out, (bs, tdim, self.nbhead * self.vdim))
        return out



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
*/
export class LayerMaskedAttention extends tf.layers.Layer {
  constructor(args) {
    super({});
    this.nbhead = args.nbhead;
    this.kdim = args.kdim;
    this.vdim = args.vdim;
    this.localAttention = args.localAttention;
  }
  computeOutputShape(inputShapes) {
    //console.log("LayerMaskedAttention computeOutputShape");
    //console.log(inputShapes);
    var inputShape = inputShapes[0];
    var outputShape = [inputShape[0], inputShape[2], this.vdim * this.nbhead];
    //console.log("outputShape ");
    //console.log(outputShape);
    return outputShape;
  }
  /*
  def attention_mask(nd, ns, localAttention,*, dtype):
      """1's in the lower triangle, counting from the lower right corner.
      Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
      """
      i = tf.range(nd)[:,None]
      j = tf.range(ns)
      m = i >= j - ns + nd
      m = tf.Print(m, [m[0:20, 0:20]], "m", summarize=1000)
      if( localAttention > 0):
          oldm = i < j - ns + nd + localAttention
          oldm = tf.Print(oldm,[oldm[0:20,0:20]],"oldm",summarize=1000 )
          m = tf.logical_and( m, oldm)
          m = tf.Print(m, [m[0:20, 0:20]], "finalm", summarize=1000)
      return tf.cast(m, dtype)
  */
  attention_mask(nd, ns, localAttention, dtype) {
    /*
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    */
    var i = tf.expandDims(tf.range(0, nd), 1);
    var j = tf.range(0, ns);
    //var m = i >= j - ns + nd;
    var m = tf.greaterEqual(i, tf.add(tf.sub(j, ns), nd));
    if (localAttention > 0) {
      var oldm = tf.less(i, tf.add(tf.sub(j, ns), tf.add(nd, localAttention)));
      m = tf.logicalAnd(m, oldm);
    }
    //console.log("mask : ");
    //console.log(m);
    return tf.cast(m, dtype);
  }
  mask_attn_weights(w, localAttention) {
    //# w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    var nd = w.shape[2];
    var ns = w.shape[3];
    var b = this.attention_mask(nd, ns, localAttention, w.dtype);
    b = tf.reshape(b, [1, 1, nd, ns]);
    w = tf.sub(tf.mul(w, b), tf.mul(tf.cast(1e10, w.dtype), tf.sub(1, b)));
    return w;
  }
  selfMaskedAttention(q, k, v, localAttention) {
    //q shape : (bs,nbhead, tdim, keydim)
    //k shape : (bs,nbhead, tdim, keydim)
    //v shape : (bs,nbhead, tdim, valdim)
    //out shape ! (bs, tdim, nbhead,valdim)
    var w = tf.matMul(q, k, false, true);
    var dimv = v.shape[3];
    w = tf.mul(w, tf.rsqrt(tf.cast(dimv, w.dtype)));
    w = this.mask_attn_weights(w, localAttention);
    //w = tf.Print(w, [tf.shape(w)],"shape w",summarize=10)
    var sm = tf.softmax(w);
    var out = tf.matMul(sm, v);
    out = tf.transpose(out, [0, 2, 1, 3]);
    return out;
  }
  call(inputs, kwargs) {
    var input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }
    //console.log("LayerMaskedAttention call");
    this.invokeCallHook(inputs, kwargs);
    const inputShape = input.shape;
    var outputShape = [inputShape[0], inputShape[2], this.vdim * this.nbhead];
    //console.log(outputShape);
    var q = inputs[0];
    var k = inputs[1];
    var v = inputs[2];
    var out = this.selfMaskedAttention(q, k, v, this.localAttention);
    out = tf.reshape(out, outputShape);
    return out;
  }
  /**
   * Layers must implement "getClassName".
   */
  getClassName() {
    return 'LayerMaskedAttention';
  }
}

LayerMaskedAttention.className = 'LayerMaskedAttention'; // static variable
