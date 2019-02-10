import * as tf from '@tensorflow/tfjs';

import { RollingImage } from './RollingImage';
import { streamDownSampler } from './streamDownSampler';

import Vue from 'vue'


//Serve Static model files using http-server -S -p 3000 --cors -C ../cert.pem -K ../key.pem 

var allAudioRaw = [];
var allAudio = [];
var allSpecter = [];
var allPr = [];

var nbOfTimeToKeep = 150;

var AudioContext = window.AudioContext // Default
  || window.webkitAudioContext // Safari and old versions of Chrome
  || false;



var globalStream = null;

function startListening() {
  //document.getElementById("output").innerText += "startListening";

  if (globalStream == null) return;
  var context = new AudioContext({
    latencyHint: 'interactive'
  });
  var source = context.createMediaStreamSource(globalStream);
  var filters = [];
  var nbfilter = 5;
  for (var i = 0; i < nbfilter; i++) {
    var f = context.createBiquadFilter();
    f.type = "lowpass";
    // To reduce the aliasing we reduce the frequency above half the target sampleFrequency of 16Khz
    // The cut off frequency is 6000 so that any frequency above 8000 are reduced by 2000/6000 * nbfilter * 12dB
    f.frequency.value = 6000;
    f.Q.value = 0;
    f.gain.value = 0;
    filters.push(f);
  }

  source.connect(filters[0]);
  for (var i = 0; i < nbfilter - 1; i++) {
    filters[i].connect(filters[i + 1]);
  }
  //document.getElementById("output").innerText += "after filter";
  var downsampler = context.createScriptProcessor(1024, 1, 1);
  //var downsampler = context.createJavaScriptNode(2048,1,1);
  filters[nbfilter - 1].connect(downsampler);
  downsampler.connect(context.destination);

  //document.getElementById("output").innerText += downsampler;
  downsampler.onaudioprocess = function (e) {
    //document.getElementById("output").innerText += "onaudioProcess";
    // Do something with the data, i.e Convert this to WAV
    var data = e.inputBuffer.getChannelData(0).slice();
    allAudioRaw.push(data);
    if (allAudioRaw.length - nbOfTimeToKeep > 0)
      allAudioRaw[allAudioRaw.length - nbOfTimeToKeep - 1] = null;
    //document.getElementById("output").innerText = allAudio.length;
    //computeSpectogram();
    sampler.process(data, e.inputBuffer.sampleRate);
  }
  document.getElementById("start").hidden = true;

}

document.getElementById("start").onclick = startListening;

var handleSuccess = function (stream) {
  //document.getElementById("output").innerText += "handleSuccess";
  globalStream = stream;

};

Math.clip = function (number, min, max) {
  return Math.max(min, Math.min(number, max));
}

var hanncst = new Array(1024);
for (var i = 0; i < 1024; i++) {
  hanncst[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / 1024);
}

var hanning = tf.tensor2d([hanncst]);


var sampler = new streamDownSampler(1024, 16000, allAudio, nbOfTimeToKeep, addComputations);

var compqueue = [];
var prqueue = [];

/*
<div>
                <input type="checkbox" id="highcompute" name="highcompute" checked>
                <label for="highcompute">High compute</label>
        </div>
*/

function addComputations() {
  //console.log("addComputations");

  //if (document.getElementById("highcompute").checked) {
  compqueue.push({ ind: allAudio.length - 2, overlap: 0.25 });
  compqueue.push({ ind: allAudio.length - 2, overlap: 0.5 });
  compqueue.push({ ind: allAudio.length - 2, overlap: 0.75 });
  //}
  compqueue.push({ ind: allAudio.length - 1, overlap: 0 });

  //compqueue.push([{ ind: allAudio.length - 1, overlap: 0 }]);
}



var gpuLock = false;
var queuesAndProcessor = [{ q: compqueue, p: computeSpectogram, skipifgreater: 150, minlength: 0 }, { q: prqueue, p: computePianoRoll, minlength: 0 }]

function computeLoop3() {
  if (gpuLock == true) {
    setTimeout(computeLoop3, 10);
    return;
  }

  var qp = null;
  var ql = -1;
  //We pick a task from the longest queue
  for (var i = 0; i < queuesAndProcessor.length; i++) {
    if (queuesAndProcessor[i].q.length - queuesAndProcessor[i].minlength > ql) {
      qp = queuesAndProcessor[i];
      ql = queuesAndProcessor[i].q.length - queuesAndProcessor[i].minlength;
    }
  }

  if (qp.skipifgreater != undefined && qp.q.length > qp.skipifgreater) {
    console.log("Skipping all past computations");
    //qp.q.length = 0;
    for( var i = 0 ; i < queuesAndProcessor.length ; i++ )
    {
      queuesAndProcessor[i].q.length = 0;
    }
    //qp.q.length = 0;
  }


  var elts = [];
  while (elts.length < 20) {
    var elt = qp.q.shift();
    if (elt == undefined) {
      break;
    }
    else {
      elts.push(elt);
    }
  }

  if (elts.length > 0) {
    //setTimeout(computeSpectogram, 0, elts);
    qp.p(elts);
    //computeSpectogram(elts);
  }

  setTimeout(computeLoop3, 10);
}


const initialStateVariable = tf.variable( tf.zeros([1,200]) );

function computePianoRoll(indices) {
  //console.log("ComputePianoRoll : ");
  //console.log( indices );


  var extraAudio = 0;


  var lastAudio = [];

  for (var i = 0; i < extraAudio; i++) {
    var ind = indices[0] - extraAudio + i;
    if (ind >= 0) {
      lastAudio.push(allSpecter[ind])
    }
    else {
      lastAudio.push(new Array(513));
    }
  }

  for (var i = 0; i < indices.length; i++) {
    lastAudio.push(allSpecter[indices[i]]);
  }

  const prfeats = tf.tidy(() => {
    var x = tf.expandDims(tf.tensor2d(lastAudio), 0);
    var inputs = [x];
    for( var i = 0; i < prmodel.initialStates.length;i++)
    {
      //inputs.push( tf.zeros([1,200]));
      //inputs.push( tf.add( initialStateVariable, tf.zeros([1,200])));
      inputs.push( prmodel.initialStates[i]);
    }
    //console.log("prmodel.model.outputs :");
    //console.log( prmodel.model.outputs);
    var results = prmodel.model.predict(inputs);
    
    if( prmodel.model.outputs.length == 1)
    {
      var logit = results;
    }
    else
    {
      var logit = results[0];
    }
    
    var greedy = tf.squeeze(tf.sigmoid(logit), 0);

    var outs = [];
    outs.push(greedy);
    for( var i= 1 ;i < results.length ; i++)
    {
      //outs.push(results[i]);
      initialStateVariable.assign( results[i]);
    }
    //outs.push( initialStateVariable);
    return outs;
    //return greedy;
  });

  gpuLock = true;

  var allpromises = Promise.all( prfeats.map( x=>x.data()) );
  //prfeats.data()
  allpromises.then( lss => {
    //console.log(lss);
    //console.log( initialStateVariable );
    var ls = lss[0];

    //console.log( lss[1]);
    //var ls = datas[0];
    for (var i = extraAudio; i < prfeats[0].shape[0]; i++) {
      allPr.push(ls.slice(i * prfeats[0].shape[1], (i + 1) * prfeats[0].shape[1]));
      if (allPr.length - nbOfTimeToKeep > 0)
        allPr[allPr.length - nbOfTimeToKeep - 1] = null;
    }
    prfeats[0].dispose();
    gpuLock = false;
  });

}



function computeSpectogram(audioIndices) {
  var lastAudio = [];
  for (var i = 0; i < audioIndices.length; i++) {
    var audioIndex = audioIndices[i];
    if (audioIndex.ind < 0)
      continue;
    if (audioIndex.overlap == 0) {
      lastAudio.push(allAudio[audioIndex.ind]);
    }
    else {
      var l = allAudio[audioIndex.ind].length;
      var buf1 = allAudio[audioIndex.ind].slice(l * audioIndex.overlap);
      var buf2 = allAudio[audioIndex.ind + 1].slice(0, l * audioIndex.overlap);
      var audio = buf1.concat(buf2);
      lastAudio.push(audio);
    }

  }

  if (lastAudio.length == 0) {
    return;
  }

  const logspecter = tf.tidy(() => {
    var x = tf.tensor2d(lastAudio);
    var fft = tf.rfft(tf.mul(x, hanning));
    var r = tf.real(fft);
    var i = tf.imag(fft);
    var logfft = tf.log(tf.add(1e-10, tf.add(tf.mul(r, r), tf.mul(i, i))));

    return logfft;
  });

  gpuLock = true;
  logspecter.data().then(ls => {
    //console.log(ls);
    for (var i = 0; i < logspecter.shape[0]; i++) {
      //if (document.getElementById("highcompute").checked) {
      allSpecter.push(ls.slice(i * logspecter.shape[1], (i + 1) * logspecter.shape[1]));
      prqueue.push(allSpecter.length - 1);
      if (allSpecter.length - nbOfTimeToKeep > 0)
        allSpecter[allSpecter.length - nbOfTimeToKeep - 1] = null;
      //}
      /*
      else {
        for (var j = 0; j < 4; j++) {
          allSpecter.push(ls.slice(i * logspecter.shape[1], (i + 1) * logspecter.shape[1]));
          prqueue.push(allSpecter.length - 1);
          if (allSpecter.length - nbOfTimeToKeep > 0)
            allSpecter[allSpecter.length - nbOfTimeToKeep-1] = null;
        }
      }
      */
    }

    /*
    for( var i = 0 ; i < ls.length ; i++)
    {
      allSpecter.push( ls[i]);
    }*/

    logspecter.dispose();
    gpuLock = false;
    //document.getElementById("output").innerText = ls;
    //drawSpectogram();

  });

}


var canvas = document.getElementById("mycanvas");
var ctx = canvas.getContext("2d");

var prcanvas = document.getElementById("pianoroll");
export var prctx = prcanvas.getContext("2d");



var colorMapRGB = [
  { r: 255, g: 0, b: 0 }, //do
  { r: 200, g: 55, b: 0 }, //do#
  { r: 150, g: 150, b: 0 },//re
  { r: 50, g: 150, b: 0 }, //re#
  { r: 0, g: 255, b: 0 },// mi
  { r: 50, g: 150, b: 55 },//fa
  { r: 0, g: 100, b: 150 },//fa#
  { r: 0, g: 0, b: 255 },//sol
  { r: 50, g: 0, b: 200 },//sol#
  { r: 100, g: 0, b: 150 },//la
  { r: 150, g: 0, b: 150 },//la@
  { r: 200, g: 50, b: 50 },//si
];

var colorMapRBG = [
  { r: 255, b: 0, g: 0 }, //do
  { r: 200, b: 55, g: 0 }, //do#
  { r: 150, b: 150, g: 0 },//re
  { r: 50, b: 150, g: 0 }, //re#
  { r: 0, b: 255, g: 0 },// mi
  { r: 50, b: 150, g: 55 },//fa
  { r: 0, b: 100, g: 150 },//fa#
  { r: 0, b: 0, g: 255 },//sol
  { r: 50, b: 0, g: 200 },//sol#
  { r: 100, b: 0, g: 150 },//la
  { r: 150, b: 0, g: 150 },//la@
  { r: 200, b: 50, g: 50 },//si
];


const wwidth = window.outerWidth || screen.width ;

prcanvas.width = Math.floor( wwidth*0.75);
canvas.width = Math.floor( wwidth *0.75);

var pianoRollRenderer = new RollingImage(prctx, prcanvas.width, 4 * 128, allPr, 2,
  function (feat, i, h) {
    var featInd = Math.floor((h - i) / 4.0);
    var v = feat[featInd];
    var colInd = (featInd + 12 - uidata.tonality)% 12;
    var col = colorMapRBG[colInd];

    return {
      r: Math.clip(Math.round(col.r * v), 0, 255),
      g: Math.clip(Math.round(col.g * v), 0, 255),
      b: Math.clip(Math.round(col.b * v), 0, 255),
      a: 255
    };
  });





var sprectoRollRenderer = new RollingImage(ctx, canvas.width, 512, allSpecter, 2,
  function (feat, i, h) {
    var v = feat[h - i];
    return {
      r: Math.clip(185 + Math.round(10 * v), 0, 255),
      g: 0, b: 0, a: 255
    };
  });





//drawSpectogram2();

pianoRollRenderer.draw();
sprectoRollRenderer.draw();

//computeLoop();
//computeLoop2();
computeLoop3();

navigator.mediaDevices.getUserMedia({ audio: true, video: false })
  .then(handleSuccess)


var uidata = {
  tonality: 0,
  mode: 'Major',
  genre: "Classical",
  modelname: "randomPolyphonic_train_10000"
};

var myview = new Vue({
  el: '#vuecontainer',
  data: uidata,
  methods: {
    onModelChange(event) {
        console.log(event.target.value);
        loadSelectedModel(event.target.value);
    }
}
});

var prmodel = null;
//randomMonophonic_train_10000
//randomPolyphonic_train_10000

function loadSelectedModel( modelname ) {
  console.log("loadSelectedModel");
  
  //var host = 'https://127.0.0.1:3000/';
  var host = "/";

  tf.loadModel(host+modelname+'/model.json').then(function (model) {
    console.log("Model loaded : "+ modelname)
    console.log(model);
    if ( prmodel != null)
    {
      prmodel.model.dispose();
    }

    var initialStates = [];
    for( var i = 1 ; i < model.inputs.length ; i++)
    {
      var shape = [1].concat( model.inputs[i].shape.slice(1));
      initialStates.push( tf.variable( tf.zeros(shape,model.inputs[i].dtype) ) );
    }

    prmodel = {"model": model,"initialStates":initialStates };
    //const xs = tf.ones([1, 30, 513]);
    //console.log(xs);
    //var res = model.predict(xs);
    //var res = tf.scalar(3.3).square();

    //document.getElementById("output").innerText = res;

    //console.log(res);
  }
  );

}


loadSelectedModel(uidata.modelname);


