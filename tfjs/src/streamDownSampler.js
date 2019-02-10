export class streamDownSampler {
  constructor(outSize, outFrequency, ouputArrays,nbOfTimeToKeep, callBack) {
    this.currentBuffer = Array(outSize);
    this.curInd = 0;
    this.currentInTime = 0.0;
    this.currentOutTime = 0.0;
    this.outdt = 1.0 / outFrequency;
    this.ouputArrays = ouputArrays;
    this.outSize = outSize;
    this.callBack = callBack;
    this.prevValue = 0;
    this.nbOfTimeToKeep = nbOfTimeToKeep;
  }
  processSample(value, indt) {
    if (this.currentInTime >= this.currentOutTime) {
      var wx = (this.currentInTime - this.currentOutTime) / indt;
      var val = wx * this.prevValue + (1 - wx) * value;
      this.currentBuffer[this.curInd] = val;
      this.currentOutTime += this.outdt;
      this.curInd++;
      if (this.curInd >= this.outSize) {
        this.ouputArrays.push(this.currentBuffer.slice());
        if (this.ouputArrays.length - this.nbOfTimeToKeep > 0)
          this.ouputArrays[this.ouputArrays.length - this.nbOfTimeToKeep - 1] = null;

        this.curInd = 0;
        this.callBack();
      }
    }
    this.currentInTime += indt;
    this.prevValue = value;
  }
  process(buffer, sampleRate) {
    //console.log(sampleRate);
    //We need to had low pass filter and interpolation
    var indt = 1.0 / sampleRate;
    for (var i = 0; i < buffer.length; i++) {
      this.processSample(buffer[i], indt);
    }
  }
}