# -*-coding:utf-8-*_

from collections import namedtuple


Label = namedtuple(
    'Label', ['id', 'categoryId', 'label', 'color', 'name'])

mask_labels = {
    Label(0,    0,      "0",    (32, 128, 192),     u"警告标志"),
    Label(1,    1,      "1",    (255, 0, 0),        u"禁令标志"),
    Label(2,    2,      "2",    (255, 255, 0),      u"指示标志"),
    Label(3,    3,      "3",    (0, 0, 255),        u"高速公路指路标志"),
    Label(4,    4,      "4",    (153, 102, 51),     u"普通道路指路标志"),
    Label(5,    5,      "5",    (102, 102, 0),      u"旅游区标志"),
    Label(6,    6,      "6",    (0, 255, 102),      u"辅助标志"),
    Label(7,    7,      "7",    (0, 102, 0),        u"作业区标志"),
    Label(8,    8,      "8",    (204, 255, 0),      u"其他标志"),
    Label(9,    9,      "9",    (255, 153, 255),    u"非交通标志"),
    Label(10,   10,     "10",   (204, 102, 102),    u"其它"),
    Label(11,   11,     "11",   (0, 0, 0),          u"Ignore")
}