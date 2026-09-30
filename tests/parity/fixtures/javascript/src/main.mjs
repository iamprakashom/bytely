import Shape, { add, mul as times } from './math.js';
import * as m from './math';

export const run = () => {
  add(1, 2);
  times(3, 4);
  return m.add(5, 6);
};

export function outer() {
  const next = () => run();
  function inner() {
    return next();
  }
  return inner();
}

export function handler(req, next) {
  return next();
}

export class Circle extends Shape {}
