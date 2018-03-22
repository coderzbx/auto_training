#!/usr/bin/env python
#encoding=utf-8

from collections import namedtuple

Label = namedtuple(
    'Label', ['className', 'alias', 'categoryId', 'trainId', 'color'])

kd_traffic_sign_labels = {
    Label('背景标志',         "background", "__background__",  0, (0, 0, 0)),
    Label('警告标志',         "warning",      "0",        1, (220, 220, 0)),
    Label('禁令标志',         "prohibition",  "1",        2, (220, 220, 0)),
    Label('指示标志',         "instructive",  "2",        3, (220, 220, 0)),
    Label('高速公路指路标志', "highway",      "3",        4, (220, 220, 0)),
    Label('普通道路指路标志', "directing",    "4",        5, (220, 220, 0)),
    Label('旅游区标志',       "travel",       "5",        6, (220, 220, 0)),
    Label('辅助标志',         "assitant",     "6",        6, (220, 220, 0)),
    Label('作业区标志',       "working",      "7",        6, (220, 220, 0)),
    Label('其他标志',         "other-sign",   "8",        6, (220, 220, 0)),
    Label('非交通标志',       "other",        "9",        6, (220, 220, 0)),
    Label('其它',             "other",        "10",       6, (220, 220, 0)),
    Label('Ignore',           "ignore",       "11",       255, (0, 0, 0)),
}

