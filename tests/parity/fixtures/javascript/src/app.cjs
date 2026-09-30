var app = exports = module.exports = {};

app.init = function init() {
  this.set('ready', true);
};

app.set = function set(setting, value) {
  return value;
};

function View(name) {
  this.name = name;
}

View.prototype.render = function render() {
  return this.lookup();
};

View.prototype.lookup = () => 'view';

exports.create = function () {
  return new View('x');
};
