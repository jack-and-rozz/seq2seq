#coding: utf-8

class DatasetBase(object):
  @property
  def size(self):
    train_size = self.train.size if hasattr(self.train, 'size') else 0
    valid_size = self.valid.size if hasattr(self.valid, 'size') else 0
    test_size = self.test.size if hasattr(self.test, 'size') else 0
    return train_size, valid_size, test_size

