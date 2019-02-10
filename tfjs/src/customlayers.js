import * as tf from '@tensorflow/tfjs';

/*
class CustomLayer1(tf.keras.layers.Layer):
    def __init__(self,):
        super(CustomLayer1, self).__init__()


    def build(self, input_shape):
        self.val = self.add_weight("value",dtype=tf.float32,shape=[1],initializer=tf.random_normal_initializer())

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        out = input + tf.reshape(self.val,(1,1,1,1))
        return out

*/


class MyCustomLayer extends tf.layers.Layer {
    constructor() {
      super({});
      // TODO(bileschi): Can we point to documentation on masking here?
      this.supportsMasking = true;
    }
    build(inputShape) {
      this.val = this.addWeight("value",[1],"float32",tf.initializers.randomNormal);
      console.log("Displaying MyCustomLayer weight");
      console.log(this.val);

      this.built=true;
    }
  
    computeOutputShape(inputShape) {
      return inputShape
    }
 
    call(inputs, kwargs) {
      var input = inputs;
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(inputs, kwargs);
      return tf.add( this.val.read().reshape([1,1,1,1]),input);
    }
  
    getClassName() {
      return 'MyCustomLayer';
    }

    /*
    getConfig() {
      const config = {};
      const baseConfig = super.getConfig();
      Object.assign(config, baseConfig);
      return config;
  }
     
    */
  }


  export function mycustomlayer() {
    //return null;
    return new MyCustomLayer();
  }

  MyCustomLayer.className = 'MyCustomLayer'; // static variable
  //tf.serialization.SerializationMap.register(MyCustomLayer);
  tf.serialization.registerClass(MyCustomLayer);
  
