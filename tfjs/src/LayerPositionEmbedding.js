import * as tf from '@tensorflow/tfjs';
/*

class LayerPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, d):
        super(LayerPositionEmbedding, self).__init__()
        self.d = d

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0],input_shape[1],self.d)

    def call(self, x):
        return positionEncodding(x[0],x[1],self.d)
*/
export class LayerPositionEmbedding extends tf.layers.Layer {
  constructor(args) {
    //console.log("LayerPositionEmbedding Constructor");
    //console.log(args);
    super({});
    // TODO(bileschi): Can we point to documentation on masking here?
    this.d = args.d;
  }
  computeOutputShape(inputShapes) {
    var inputShape = inputShapes[0];
    //console.log("LayerPositionEmbedding computeOutputShape");
    //console.log(inputShapes);
    //console.log("LayerPositionEmbedding layer :")
    //console.log(this);
    var outputShape = [inputShape[0], inputShape[1], 2 * this.d];
    //console.log(outputShape);
    return outputShape;
  }
  /*
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
  */
  positionEncodding(x, nbpast, d) {
    var bs = x.shape[0];
    var tdim = x.shape[1];
    var offsetpast = tf.cast(tf.reshape(nbpast, [bs, 1, 1]), "float32");
    var rshprange = tf.reshape(tf.range(0, tdim, 1, "float32"), [1, Math.floor(tdim), 1]);
    var pos = tf.add(offsetpast, rshprange);
    var featrange = tf.reshape(tf.pow(10000.0, tf.div(tf.range(0, d, 1, "float32"), d)), [1, 1, d]);
    var div = tf.div(pos, featrange);
    var cos = tf.cos(div);
    var sin = tf.sin(div);
    return tf.concat([cos, sin], 2);
  }
  call(inputs, kwargs) {
    this.invokeCallHook(inputs, kwargs);
    const inputShape = inputs[0].shape;
    //console.log("LayerPositionEmbedding call");
    //console.log(inputShape);
    return this.positionEncodding(inputs[0], inputs[1], this.d);
  }
  /**
   * Layers must implement "getClassName".
   */
  getClassName() {
    return 'LayerPositionEmbedding';
  }
}

LayerPositionEmbedding.className = 'LayerPositionEmbedding'; // static variable


