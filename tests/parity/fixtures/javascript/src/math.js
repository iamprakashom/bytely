export const LIMIT = 8;
let counter = 0;

export function add(a, b) {
  return a + b;
}

export const mul = (a, b) => a * b;

function* ids() {
  yield counter++;
}

export default class Shape {
  constructor(size) {
    this.size = size;
  }
  area() {
    return this.scale(this.size);
  }
  scale(n) {
    return mul(n, 2);
  }
  static unit() {
    return new Shape(1);
  }
}

export class Square extends Shape {
  area() {
    return this.scale(1);
  }
}
