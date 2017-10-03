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
      result.values[row] += self.values[ix] * vector.values[col]
    }
  }

}

MatrixMN.prototype.multiplyTransposed = function (vector, result) {
  var self = this

  console.assert(self.numRows <= result.length)
  console.assert(self.numCols <= vector.length)

  for (var col = 0; col < self.numRows; col++) {

    result[col] = 0

    for (var row = 0; row < self.numCols; row++, ix += self.numCols) {

      result.values[col] += self.values[ix] * vector.values[row]
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