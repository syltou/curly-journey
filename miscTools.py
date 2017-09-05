# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:47:20 2017

@author: Sylvain
"""

import random, string




def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

