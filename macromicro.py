def TP(self, classe):
      '''Returns the true positive count for class classe'''
      if not self.classes:
          raise EmptyError()
      try:
          return self._classes[classe][classe]
      except KeyError:
          return 0


  def FP(self, classe):
      '''Returns the false positive count for class classe'''
      if not self.classes:
          raise EmptyError()
      if classe not in self.classes:
          return 0
      fp=0
      for k,v in self._classes.items():
          if k != classe:
              fp+=v.get(classe, 0)
      return fp

  def FN(self, classe):
      '''Returns the false negative count for class classe'''
      if not self.classes:
          raise EmptyError()
      if classe not in self.classes:
          return 0
      fn = 0
      for k,v in self._classes[classe].items():
          if k != classe:
              fn+=v
      return fn

  def TN(self, classe):
      '''Returns the true negative count for class classe'''
      if not self.classes:
          raise EmptyError()
      if classe not in self.classes:
          raise ClassError(classe)
      return len(self) - self.TP(classe) - self.FP(classe) - self.FN(classe)

  def precision(self, classe):
      '''Return the precision of the class classe'''
      tp = float(self.TP(classe))
      fp= self.FP(classe)

      try:
          return tp/(tp+fp)
      except ZeroDivisionError:
          return 0

  def recall(self, classe):
      '''Returns the recall of the class classe'''
      tp = float(self.TP(classe))

      if tp == 0:
          return 0

      fn= self.FN(classe)
      return tp/(tp+fn)

  def fmeasure(self, classe, beta=1.0):
      '''Returns the fmeasure with beta for the class classe:
      (1.0 + beta) * (P*R) / (beta*P + R)'''
      P = self.precision(classe)
      R = self.recall(classe)
      try:
          return (1.0 + beta) * (P*R) / (beta*P + R)
      except ZeroDivisionError:
          return 0

  def macrofmeasure(self, beta=1.0):
      '''The macroaveraged fmeasure
      '''
      if not isinstance(beta, float):
          raise TypeError('beta must be a float or integer')

      try:
          return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.training)
      except TrainingError:
          return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.classes)


  def microfmeasure(self, beta=1.0):
      '''The microaveraged fmeasure.
      '''
      if not isinstance(beta, float):
          raise TypeError('beta must be a float or integer')

      try:
          return sum([self.fmeasure(c, beta=beta)*self.training[c]/sum(self.training.values()) for c in self.training.keys()])
      except TrainingError:
          return sum([self.fmeasure(c, beta=beta)*self.count(c)/len(self) for c in self.classes])
      
