import re
import array
import numpy as np


class SimpleTokenizer(object):

    def tokenize(self, mass):
        pattern = re.compile(u'[a-zA-Z].*')
        return [x.strip() for x in re.split(' |\n|\t|\)|\.', mass) if pattern.match(x) != None]
