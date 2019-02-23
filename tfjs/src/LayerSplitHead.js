import * as tf from '@tensorflow/tfjs';
/*

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

*/
export class LayerSplitHead extends tf.layers.Layer {
  constructor(args) {
    super({});
    this.nbhead = args.nbhead;
    this.outdim = args.outdim;
  }
  computeOutputShape(inputShape) {
    //console.log("LayerSplitHead computeOutputShape");
    //console.log(inputShape);
    return [inputShape[0], this.nbhead, inputShape[1], this.outdim];
  }
  call(inputs, kwargs) {
    var input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }
    //console.log("LayerSplitHead call");
    this.invokeCallHook(inputs, kwargs);
    const inputShape = input.shape;
    var bs = inputShape[0];
    var tdim = inputShape[1];
    var q = tf.reshape(input, [bs, tdim, this.outdim, this.nbhead]);
    var out = tf.transpose(q, [0, 3, 1, 2]);
    //const outShape = [inputShape[0], this.nbhead, inputShape[1], this.outdim];
    //console.log(outShape);
    //return tf.zeros(outShape);
    return out;
  }
  /**
   * Layers must implement "getClassName".
   */
  getClassName() {
    return 'LayerSplitHead';
  }
}

LayerSplitHead.className = 'LayerSplitHead'; // static variable

