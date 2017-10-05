function MatrixMN () {
  var self = this

  self.numRows = 0
  self.numCols = 0

  self.values = []

}

MatrixMN.prototype.init = function (row, col, initial = true) {
  var self = this

  var numAll_old = self.numRows * self.numCols

  self.numRows = row
  self.numCols = col

  var numAll = self.numRows * self.numCols

  if (numAll === numAll_old) {

    if (self.numRows * self.numCol > Number.MAX_SAFE_INTEGER) throw new Error('integer overflow')

    // matrix의 크기만큼 0으로 초기화한다.
    if (initial) self.values.fill.call({length: numAll}, 0)
  }
}
/**
 * 둘 다 원래의 소스에서는 vector 객체이다
 * 나 자신과 input 행렬의 곱을 result에 주는 것 이다
 * 둘 다 행렬이어야 한다
 * @param vector
 * @param result
 */
MatrixMN.prototype.multiply = function (vector, result) {
  var self = this

  console.assert(self.numRows <= result.length)
  console.assert(self.numCols <= vector.length)

  for (var row = 0; row < self.numRows; row++) {

    result[row] = 0
    var ix = row * self.numCols

    for (var col = 0; col < self.numCols; col++, ix++) {
      // 나의 행렬값과 vector의 행렬 값을 곱한 것들을 result에 더한다
      result[row] += self.values[ix] * vector[col]

    }
  }

}

MatrixMN.prototype.multiplyTransposed = function (vector, result) {
  var self = this

  console.assert(self.numRows <= vector.length)
  console.assert(self.numCols <= result.length)

  for (var col = 0; col < self.numCols; col++) {

    result[col] = 0

    for (var row = 0, ix = col; row < self.numRows; row++, ix += self.numCols) {

      result[col] += self.values[ix] * vector[row]
      //console.log(JSON.stringify(self.values[ix]))
    }
  }
}

MatrixMN.prototype.toString = function () {
  var self = this
  // 크롬에서 어차피 정리되어 나온다
  console.log(self.values)
}

MatrixMN.prototype.get1DIndex = function (row, col) {
  var self = this

  console.assert(row >= 0)
  console.assert(col >= 0)
  console.assert(row < self.numRows)
  console.assert(row < self.numCols)

  return col + row * self.numRows
}

MatrixMN.prototype.getValue = function (row, col) {
  var self = this

  return self.values[self.get1DIndex(row, col)]
}

function NeuralNetwork () {
  var self = this
  //input layer의 입력이 몇 개냐
  self.numInput = 0.0
  //output layer의 출력이 몇개냐
  self.numOutput = 0.0
  // 액티베이션 함수가 있는 뉴런들(hidden layer) + 입력,출력 레이어(2)
  self.numAllLayers = 0.0

  // bias는 고정적이고 bias의 weight값을 조정 함으로써 bias를 조정하는 방식으로 구현이 되어있다
  self.bias = 0.0
  // learning rate
  self.alpha = 0.0

  // 2차원으로 구성 될 것이다
  // 배열로 구성되는 이유는 레이어가(뉴런) 여러개일 수 있기 때문이다
  // js에서는 배열이 벡터와 유사함으로 2차원 배열로 구현해도 무방했다
  // 액티베이션 된 후의 값을 저장할 것 이다
  self.layerNeuronAct = [] // layerNeuronAct[0] = input layer,  layerNeuronAct[numAllLayers - 1] = output layer, layerNeuronAct[i][j] = activation value
  self.layerNeuronGrad = [] // back propagation에서 사용되는 gradient 값이 저장된다
  self.weights = []

  self.numLayerActs = [] // 각 layer의 activation value의 개수를 의미한다 bias까지 포함되어어있다
}

/**
 *
 * @param numInput
 * @param numOutput
 * @param numHiddenLayers : 숨겨진 layer의 개수를 의미란다
 */
// 이 객체는 각 층이 한개의 뉴런으로 되어있는 모델을 생각하고 만든 것 같다
NeuralNetwork.prototype.init = function (numInput, numOutput, numHiddenLayers) {
  var self = this
  // 모든 원소에 input의 개수를 default 값으로 넣어준다
  self.numLayerActs = [].fill.call({length: numHiddenLayers + 2}, numInput + 1)

  // 0번째 layer는 input layer이다
  self.numLayerActs[0] = numInput + 1
  // numHiddenLayers + 1번째 레이어는 output layer 이다
  self.numLayerActs[numHiddenLayers + 1] = numOutput + 1

  self.numInput = self.numLayerActs[0] - 1
  self.output = self.numLayerActs[numHiddenLayers + 1] - 1
  self.numAllLayers = numHiddenLayers + 2

  self.bias = 1.0
  self.alpha = 0.15

  //모든 레이어에 대해
  // 각 뉴런에 대해 추가한다.
  // 뭔가 여기 잘 이해 안된다

  for (var i = 0; i < self.numAllLayers; i++) {
    self.layerNeuronAct[i] = [].fill.call({length: self.numLayerActs[i]}, 0.0)
    self.layerNeuronAct[i][self.numLayerActs[i] - 1] = self.bias
  }

  for (var i = 0; i < self.numAllLayers; i++) {
    self.layerNeuronGrad[i] = [].fill.call({length: self.numLayerActs[i]}, 0.0)
  }

  for (var i = 0; i < self.numAllLayers - 1; i++) {
    self.weights[i] = new MatrixMN()
    self.weights[i].init(self.layerNeuronAct[i + 1].length - 1, self.layerNeuronAct[i].length)
    for (var ix = 0; ix < self.weights[i].numRows * self.weights[i].numCols; ix++) {
      self.weights[i].values[ix] = Math.random() / Number.MAX_SAFE_INTEGER * 0.1
    }
  }
}

NeuralNetwork.prototype.getSigmoid = function (x) {
  return 1.0 / ( 1.0 + Math.exp(-x))
}

NeuralNetwork.prototype.getSigmoidGradFromY = function (y) {
  return (1.0 - y) * y
}

NeuralNetwork.prototype.getRELU = function (x) {
  return Math.max(0.0, x)
}

NeuralNetwork.prototype.getRELUGradFromY = function (x) {
  return 0.0 > x ? 0.0 : 1.0
}

NeuralNetwork.prototype.getLRELU = function (x) {
  return x > 0.0 ? x : 0.01 * x
}

NeuralNetwork.prototype.getLRELUGradFromY = function (x) {
  return x > 0.0 ? 1.0 : 0.01
}

NeuralNetwork.prototype.applySigmoidToVector = function (vector) {
  var self = this

  // vector 전체를 순회하면서 Sigmoid를 각 원소에 대해 적용한다
  vector.map(function (x) { return self.getSigmoid(x) })
}

NeuralNetwork.prototype.applyRELUToVector = function (vector) {
  var self = this

  // vector 전체를 순회하면서 Sigmoid를 각 원소에 대해 적용한다
  for (var i = 0, len = vector.length; i < len; i++) {
    vector[i] = self.getRELU(vector[i])
  }
}

NeuralNetwork.prototype.applyLRELUToVector = function (vector) {
  var self = this

  // vector 전체를 순회하면서 Sigmoid를 각 원소에 대해 적용한다
  vector.map(function (x) { return self.getLRELU(x) })
}

NeuralNetwork.prototype.propForward = function () {
  var self = this
  // 각 weight에 대해서 현재의 activation 함수와 곱해서 다음 액티베이션 함수에 적용한다
  for (var i = 0, len = self.weights.length; i < len; i++) {
    // 아마 여기가 affine sum
    self.weights[i].multiply(self.layerNeuronAct[i], self.layerNeuronAct[i + 1])
    // 여기가 activation 함수 적용
    self.applyRELUToVector(self.layerNeuronAct[i + 1])
  }
}

NeuralNetwork.prototype.propBackward = function (target) {
  var self = this

  var l = self.layerNeuronGrad.length - 1

  for (var d = 0, len = self.layerNeuronGrad[l].length - 1; d < len; d++) {
    var outputValue = self.layerNeuronAct[l][d]

    // E 함수 미분한 것을 적용하는 것 이다
    self.layerNeuronGrad[l][d] = (target[d] - outputValue) * self.getRELUGradFromY(outputValue)

  }

  for (var i = self.weights.length - 1; i >= 0; i--) {
    self.weights[i].multiplyTransposed(self.layerNeuronGrad[i + 1], self.layerNeuronGrad[i])

    for (var j = 0, len = self.layerNeuronAct[i].length; j < len; j++) {
      self.layerNeuronGrad[i][j] += self.getRELUGradFromY(self.layerNeuronAct[i][j])
    }
  }

  for (var i = self.weights.length - 1; i >= 0; i--) {

    self.updateWeight(self.weights[i], self.layerNeuronGrad[i + 1], self.layerNeuronAct[i])
  }

}

NeuralNetwork.prototype.updateWeight = function (weightMatrix, nextLayerGrad, prevLayerAct) {
  var self = this

  for (var row = 0; row < weightMatrix.numRows; row++) {
    for (var col = 0; col < weightMatrix.numCols; col++) {
      var delta_w = self.alpha * nextLayerGrad[row] * prevLayerAct[col]

      weightMatrix.values[weightMatrix.get1DIndex(row, col)] += delta_w
    }
  }
}
// inpu layer의 값을 설정한다
NeuralNetwork.prototype.setInputVector = function (input) {
  var self = this

  if (input.length < self.numInput) console.log('Input dimension is wrong')

  for (var d = 0; d < self.numInput; d++) {
    self.layerNeuronAct[0][d] = input[d]
  }
}

NeuralNetwork.prototype.copyOutputVector = function (copy, copy_bias) {
  var self = this

  var outputLayerAct = self.layerNeuronAct[self.layerNeuronAct.length - 1]
  var len = (copy_bias === true ? self.numOutput : self.numOutput + 1)

  for (var i = 0; i < len; i++) {
    copy[i] = outputLayerAct[i]
  }
}

var x = [].fill.call({length: 2}, 0.0)
var y_target = [1.8, 0.3]
var y_temp = [0.0, 0.0]

var nn = new NeuralNetwork()

nn.init(2.0, 1.0, 1.0)
nn.alpha = 0.1

for (var i = 0; i < 100; i++) {
  // forward propagation에 직접 x를 넣어도 되는데 다른 라이브러리에서 이렇게 분리를 했길래 했다.
  nn.setInputVector(x)
  nn.propForward()

  nn.copyOutputVector(y_temp)
  console.log('order : ', i, ' - ', y_temp)
  //console.log(JSON.stringify(nn.weights))
  if(Math.abs(y_temp[0] - y_target[0]) < 0.01) break
  nn.propBackward(y_target)
}






