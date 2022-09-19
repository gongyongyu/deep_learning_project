# coding:utf-8

import numpy as np

'''
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]
c = [11, 12, 13, 14, 15]
d = list(zip(a, b, c))
print(d)

'''


class A:
    def add(self, x=0):
        print(x + 1)


class B(A):
    def add(self, x=0):
        super().add(x)


b = B()
b.add(2)