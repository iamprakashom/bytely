const { add, mul: product } = require('./math.js');
const lib = require('./lib');
const fs = require('fs');

const helper = function () {
  return add(1, 1) + product(2, 2) + lib.twice(3);
};

function read() {
  return fs.readFileSync('x');
}

module.exports = { helper, read };
