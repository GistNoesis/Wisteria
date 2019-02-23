import * as tf from '@tensorflow/tfjs';
/*
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
*/
export class LayerNorm extends tf.layers.Layer {
  constructor(args) {
    super({});
    this.axis = args.axis;
    //console.log("LayerNorm constructor");
    //console.log(this.axis);
  }
  computeCorrectName(name) {
    //layer_norm_LayerNorm1/g  must be converted to layer_norm/g
    //layer_norm_LayerNorm2/g  must be converted to layer_norm_1/g
    var prefix = "layer_norm_LayerNorm";
    var subsr = name.substr(prefix.length);
    var vals = subsr.split('/');
    var num = parseInt(vals[0]);
    var nbstr = num == 1 ? "" : "_" + (num - 1).toString();
    var res = "layer_norm" + nbstr + "/" + vals[1];
    return res;
  }
  build(inputShape) {
    //console.log("LayerNorm build : ")
    this.g = this.addWeight("g", [inputShape[inputShape.length - 1]], "float32", tf.initializers.ones());
    this.b = this.addWeight("b", [inputShape[inputShape.length - 1]], "float32", tf.initializers.zeros());
    var gname = this.computeCorrectName(this.g.originalName);
    var bname = this.computeCorrectName(this.b.originalName);
    this.g.originalName = gname;
    this.g.name = gname;
    this.b.originalName = bname;
    this.b.name = bname;
    //console.log(this.g);
    //console.log(this.b);
    this.built = true;
  }
  computeOutputShape(inputShape) {
    //console.log("LayerNorm computeOutputShape");
    //console.log(inputShape);
    return inputShape;
  }
  call(inputs, kwargs) {
    var input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }
    //console.log("LayerNorm call");
    this.invokeCallHook(inputs, kwargs);
    var x = input;
    var epsilon = 1e-5;
    var axis = this.axis == -1 ? input.shape.length - 1 : this.axis;
    var u = tf.mean(x, axis, true);
    var xmu = tf.sub(x, u);
    var s = tf.mean(tf.square(xmu), axis, true);
    x = tf.mul(xmu, tf.rsqrt(tf.add(s, epsilon)));
    //var gval = this.g.read();
    var gval = tf.reshape(this.g.read(), [1, 1, -1]);
    var bval = tf.reshape(this.b.read(), [1, 1, -1]);
    //x = x * + tf.reshape( this.b,[1,1,-1]);
    x = tf.add(tf.mul(x, gval), bval);
    return x;
  }
}

LayerNorm.className = 'LayerNorm'; // static variable

