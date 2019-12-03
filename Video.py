# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:51:02 2019

@author: lpc
"""

class Video:
    label_count = [0 for i in range(12)]
    def __init__(self,ad_id,url,label,label_num,count):
        self.ad_id = ad_id
        self.url = url
        self.label = label
        self.label_num = label_num
        self.count = count