
function setArrayToVal(data, r, g, b, a) {
  for (var i = 0; i < data.length; i += 4) {
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = a;
  }
}

export function createEmptyImage(ctx, w, h) {
  var img = ctx.getImageData(0, 0, w, h);
  img.data.set(new Uint8ClampedArray(w * h * 4));
  setArrayToVal(img.data, 0, 0, 0, 255);
  return img;
}

export class RollingImage {
  constructor(ctx, w, h, buffers, nbofimageToKeep, transfo) {
    this.ctx = ctx;
    this.w = w;
    this.h = h;
    this.imgs = [];
    this.lastDrawn = -1;
    this.buffers = buffers;
    this.transfo = transfo;
    this.nbofimageToKeep = nbofimageToKeep;
  }
  draw() {
    var th = this;
    requestAnimationFrame(() => th.draw());
    var ind = this.buffers.length - 1;
    if (ind < 0)
      return;
    for (var jj = this.lastDrawn + 1; jj < this.buffers.length; jj++) {
      var imgInd = Math.floor(jj / this.w);
      if (imgInd >= this.imgs.length) {
        this.imgs.push(createEmptyImage(this.ctx, this.w, 500));
        if (this.imgs.length - this.nbofimageToKeep > 0) {
          this.imgs[this.imgs.length - this.nbofimageToKeep - 1] = null;
        }

      }
      var modInd = jj % this.w;
      var data = this.imgs[imgInd].data;
      var feat = this.buffers[jj];
      var j = modInd;
      if (feat != null) {
        for (var i = 0; i < this.h; i++) {
          //var value = feat[ Math.floor( (this.h - i) / 5) ];
          //value = Math.clip( Math.round(255 * value), 0, 255);
          var value = this.transfo(feat, i, this.h);
          data[4 * i * this.w + 4 * j] = value.r;
          data[4 * i * this.w + 4 * j + 1] = value.g;
          data[4 * i * this.w + 4 * j + 2] = value.b;
          data[4 * i * this.w + 4 * j + 3] = value.a;
        }
      }
    }
    var imgInd = Math.floor(ind / this.w);
    var j = ind % this.w;
    if (imgInd > 0) {
      this.ctx.putImageData(this.imgs[imgInd - 1], -j, 0);
    }
    this.ctx.putImageData(this.imgs[imgInd], this.w - j, 0);
    this.lastDrawn = this.buffers.length - 1;
  }
}