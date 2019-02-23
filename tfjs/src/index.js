import * as tf from '@tensorflow/tfjs';

import { RollingImage } from './RollingImage';
import { streamDownSampler } from './streamDownSampler';

import Vue from 'vue'



import { LayerNorm } from './LayerNorm';
import { LayerMaskedAttention } from './LayerMaskedAttention';
import { LayerSplitHead } from './LayerSplitHead';
import { LayerPositionEmbedding } from './LayerPositionEmbedding';

tf.serialization.registerClass(LayerNorm);
tf.serialization.registerClass(LayerMaskedAttention);
tf.serialization.registerClass(LayerSplitHead);
tf.serialization.registerClass(LayerPositionEmbedding);


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
  var co = 0;
  downsampler.onaudioprocess = function (e) {

    //document.getElementById("output3").innerText = "onaudioProcess " + co;
    co = co + 1;
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
var queuesAndProcessor = [{ q: compqueue, p: computeSpectogram, skipifgreater: 60, minlength: 0 }, { q: prqueue, p: computePianoRoll, minlength: 0 }]


function computeLoop3() {
  if (gpuLock == true) {
    setTimeout(computeLoop3, 10);
    return;
  }

  //setTimeout(computeLoop3, 10);
  //return;

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
    for (var i = 0; i < queuesAndProcessor.length; i++) {
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


//const initialStateVariable = tf.variable(tf.zeros([1, 200]));

function computePianoRoll(indices) {
  //return;
  //console.log("ComputePianoRoll : ");
  //console.log( indices );
  if (prmodel == null) {
    return;
  }

  var extraAudio = 0;


  var lastAudio = [];

  for (var i = 0; i < extraAudio; i++) {
    var ind = indices[0] - extraAudio + i;
    if (ind >= 0) {
      lastAudio.push(allSpecter[ind]);
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
    //console.log("prmodel.architecture : " + prmodel.architecture);

    if (prmodel.architecture === "transformer") {
      //inputs.push( tf.zeros( [1,1], "int32") );
      inputs.push(prmodel.pastCounter);
      //console.log("adding 0 as first input");
    }

    for (var i = 0; i < prmodel.initialStates.length; i++) {
      inputs.push(prmodel.initialStates[i]);
    }

    for (var i = 0; i < inputs.length; i++) {
      //console.log( "inputs[" + i +"] shape  : " + inputs[i].shape)
    }

    //console.log("prmodel.model.outputs :");
    //console.log( prmodel.model.outputs);
    var results = prmodel.model.predict(inputs);

    if (prmodel.model.outputs.length == 1) {
      var logit = results;
    }
    else {
      var logit = results[0];
    }

    prmodel.pastCounter.assign(tf.add(prmodel.pastCounter, tf.cast(logit.shape[1], "int32")));

    var greedy = tf.squeeze(tf.sigmoid(logit), 0);

    var outs = [];
    outs.push(greedy);

    if (prmodel.architecture === "GRU") {
      for (var i = 0; i < prmodel.initialStates.length; i++) {
        prmodel.initialStates[i].assign(results[i + 1]);
      }
    }
    else if (prmodel.architecture === "transformer") {

      for (var i = 0; i < prmodel.initialStates.length; i++) {
        //var ccat = tf.concat( [prmodel.initialStates[i], results[i+1]],2 ) ;
        //console.log("ccat " + i);
        //console.log(ccat.shape);
        //var sl = tf.slice4d(ccat, [0,0,ccat.shape[2]-10,0],[-1,-1,-1,-1]);
        var nl = results[i + 1].shape[2];
        var vl = prmodel.initialStates[i].shape[2];
        if (nl < vl) {
          var sl = tf.slice4d(prmodel.initialStates[i], [0, 0, vl - nl, 0], [-1, -1, -1, -1]);
          var res = tf.concat([sl, results[i + 1]], 2);
        }
        else if (nl == vl) {
          var res = results[i + 1];
        }
        {
          var res = tf.slice4d(results[i + 1], [0, 0, nl - vl, 0], [-1, -1, -1, -1]);
        }


        //console.log("sl " + i);
        //console.log(sl.shape);
        prmodel.initialStates[i].assign(res);
      }

    }


    return outs;
  });

  gpuLock = true;//true

  var allpromises = Promise.all(prfeats.map(x => x.data()));
  //prfeats.data()
  allpromises.then(lss => {
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

  gpuLock = true;//true
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

var prmodel = null;

function whenDocumentReady() {

  document.getElementById("start").onclick = startListening;

  var canvas = document.getElementById("mycanvas");
  var ctx = canvas.getContext("2d");

  var prcanvas = document.getElementById("pianoroll");
  var prctx = prcanvas.getContext("2d");



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


  const wwidth = window.outerWidth || screen.width;

  prcanvas.width = Math.floor(wwidth * 0.75);
  canvas.width = Math.floor(wwidth * 0.75);

  var pianoRollRenderer = new RollingImage(prctx, prcanvas.width, 4 * 128, allPr, 2,
    function (feat, i, h) {
      var featInd = Math.floor((h - i) / 4.0);
      var v = feat[featInd];
      var colInd = (featInd + 12 - uidata.tonality) % 12;
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

//"randomPolyphonic_train_10000"
  var uidata = {
    tonality: 0,
    mode: 'Major',
    genre: "Classical",
    modelname: "randomPolyphonic_train_10000_AllPianos-transformer_3_100_10_-robust"
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


  //randomMonophonic_train_10000
  //randomPolyphonic_train_10000

  var loadModelLock = false;

  function loadSelectedModel(modelname) {
    console.log("loadSelectedModel");
    console.log(modelname);
    console.log(loadModelLock);
    if (loadModelLock == true) {
      return;
    }
    loadModelLock = true;
    //var host = 'https://127.0.0.1:3000/';
    var host = "/";

    tf.loadModel(host + modelname + '/model.json').then(function (model) {
      loadModelLock = false;
      console.log("Model loaded : " + modelname)
      //console.log(model);
      //document.getElementById("output").innerText="Model loaded";
      if (prmodel != null) {
        prmodel.model.dispose();
        //dispose of initialStates variables
        for (var i = 0; i < prmodel.initialStates.length; i++) {
          prmodel.initialStates[i].dispose();
          prmodel.pastCounter.dispose();
        }
      }

      var architecture = null;
      var modname = modelname.split("-")[1];
      if (modname != undefined)
      {
        var modparams = modname.split("_")
        architecture = modparams[0];
      }

      var initialStates = [];

      if (architecture === "GRU") {
        for (var i = 1; i < model.inputLayers.length; i++) {
          var shape = [1].concat(model.inputLayers[i].batchInputShape.slice(1));
          initialStates.push(tf.variable(tf.zeros(shape, model.inputs[i].dtype)));
        }
      }
      if (architecture === "transformer") {
        //console.log("model : ")
        //console.log(model);
        var causalWindow = parseInt( modparams[3] )-1;
        for (var i = 2; i < model.inputLayers.length; i++) {

          var inpshape = model.inputLayers[i].batchInputShape;//(bs=null,nbhead,tdim=null,featdim)
          //console.log("inpshape : " )
          //console.log(inpshape)
          var shape = [1, inpshape[1], causalWindow, inpshape[3]];
          initialStates.push(tf.variable(tf.zeros(shape, model.inputs[i].dtype)));
        }
      }

      var pastCounter = tf.variable(tf.zeros([1, 1], "int32"));

      prmodel = { "model": model, "initialStates": initialStates, "architecture": architecture, "pastCounter": pastCounter };
      /*
      //Debug to check that the computations gives the same value in python and in javascript
      if (architecture == "transformer") {
        const xs = tf.ones([1, 3, 513]);
        //console.log(xs);

        var inputs = [];
        inputs.push(xs);
        inputs.push(prmodel.pastCounter);
        for (var i = 0; i < initialStates.length; i++) {
          inputs.push(initialStates[i]);
        }
        var res = model.predict(inputs);
        if( Array.isArray(res) == false)
        {
          res = [res];
        }
        var allpromises = Promise.all(res.map(x => x.data()));
        allpromises.then( lss=> console.log(lss) );
      }
      */
      //var res = tf.scalar(3.3).square();

      //document.getElementById("output").innerText = res;

      //console.log(res);
    }
    );

  }

  //document.getElementById("output").innerText="Starting model load";
  loadSelectedModel(uidata.modelname);

  //loadSelectedModel("randomMonophonic_train_100_AllPianos-transformer_3_100_-robustlowmem");
  //loadSelectedModel("testTransformer");

}
window.onload = whenDocumentReady;