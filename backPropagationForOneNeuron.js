function Neuron (w, b) {
  var self = this

  self.weight = w || 2.0
  self.bias = b || 1.0

  self.output = 0.0
  self.input = 0.0
}

Neuron.prototype.getAct = function (x) {
  return Number(x)
}

Neuron.prototype.getActGrad = function (x) {
  // 지금은 함수가 x 라서 이렇게 한다
  return 1.0
}

Neuron.prototype.feedFoward = function (input) {
  // 지금은 함수가 x 라서 이렇게 한다
  var self = this

  var sigma = input * self.weight + self.bias
  self.output = self.getAct(sigma)

  return self.output
}

Neuron.prototype.propBackward = function (target) {
  // 지금은 함수가 x 라서 이렇게 한다
  var self = this
  // E 값의 미분값을 구하는 것 이다. (1/2(y_target-y_output)^2)'
  var alpha = 0.1
  var grad = (self.output - target) * self.getActGrad(self.output)

  self.weight = self.weight - alpha * grad * self.input
  self.bias = self.bias - alpha * grad * 1.0
}

Neuron.prototype.feedFowardAndPrint = function (input) {
  var self = this
  console.log("input : " , input , " " ,self.feedFoward(input))
}


var my_neuron = new Neuron(2.0, 1.0)

for(var i = 0; i < 1000; i++){
  console.log('training : ' , i)

  my_neuron.feedFowardAndPrint(1.0)
  my_neuron.propBackward(4.0)
  my_neuron.feedFowardAndPrint(1.0)
}