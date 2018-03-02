# -*-coding:utf-8-*-

from collections import namedtuple
import os
from PIL import Image

Label = namedtuple(
        'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])

self_server_label = {
    Label('other',          0,  11,     (64, 64, 32),       u'其他'),
    Label('ignore',         1,  255,    (0, 0, 0),          u'Ignore'),
    Label('lane',           2,  0,      (255, 0, 0),        u'车道标线'),
    Label('left',           3,  1,      (255, 192, 203),    u'左侧道路边缘线'),
    Label('right',          4,  2,      (139, 0, 139),      u'右侧道路边缘线'),
    Label('v_slow',         5,  3,      (32, 128, 192),     u'纵向减速标线'),
    Label('bus_lane',       6,  4,      (192, 128, 255),    u'专用车道标线'),
    Label('stop',           7,  5,      (255, 128, 64),     u'停止线'),
    Label('slow_let',       8,  5,      (0, 255, 255),      u'减速让行标线'),
    Label('slow_zone',      9,  5,      (128, 128, 255),    u'减速标线/减速带'),
    Label('sidewalk',       10, 5,      (128, 192, 192),    u'人行横道'),
    Label('connection',     11, 5,      (128, 128, 192),    u'路面连接带'),
    Label('stop_station',   12, 0,      (240, 128, 128),    u'停靠站标线'),
    Label('in_out',         13, 6,      (128, 128, 0),      u'出入口标线'),
    Label('symbol',         14, 7,      (0, 0, 255),        u'文字符号类'),
    Label('fish_lane',      15, 8,      (0, 255, 0),        u'导流线（鱼刺线）'),
    Label('stop_gird',      16, 5,      (255, 255, 0),      u'停止网格标线'),
    Label('distance',       17, 5,      (255, 128, 255),    u'车距确认线'),
    Label('road',           18, 9,      (192, 192, 192),    u'道路'),
    Label('objects',        19, 10,     (128, 0, 0),        u'车辆及路面上其他物体'),
}

Label = namedtuple(
    'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])

self_road_chn_labels = {
    Label('other',          0, 0,     (64, 64, 32),       u'其他'),
    Label('ignore',         1, 1,     (0, 0, 0),          u'Ignore'),
    Label('lane',           2, 2,     (255, 0, 0),        u'车道标线'),
    Label('left',           3, 3,     (255, 192, 203),    u'左侧道路边缘线'),
    Label('right',          4, 4,     (139, 0, 139),      u'右侧道路边缘线'),
    Label('v_slow',         5, 5,     (32, 128, 192),     u'纵向减速标线'),
    Label('bus_lane',       6, 6,     (192, 128, 255),    u'专用车道标线'),
    Label('stop',           7, 7,     (255, 128, 64),     u'停止线'),
    Label('slow_let',       8, 8,     (0, 255, 255),      u'减速让行标线'),
    Label('slow_zone',      9, 9,     (128, 128, 255),    u'减速标线/减速带'),
    Label('sidewalk',       10, 10,   (128, 192, 192),    u'人行横道'),
    Label('connection',     11, 11,   (128, 128, 192),    u'路面连接带'),
    Label('stop_station',   12, 12,   (240, 128, 128),    u'停靠站标线'),
    Label('in_out',         13, 13,   (128, 128, 0),      u'出入口标线'),
    Label('symbol',         14, 14,   (0, 0, 255),        u'文字符号类'),
    Label('fish_lane',      15, 15,   (0, 255, 0),        u'导流线（鱼刺线）'),
    Label('stop_gird',      16, 16,   (255, 255, 0),      u'停止网格标线'),
    Label('distance',       17, 17,   (255, 128, 255),    u'车距确认线'),
    Label('road',           18, 18,   (192, 192, 192),    u'道路'),
    Label('objects',        19, 19,   (128, 0, 0),        u'车辆及路面上其他物体'),
}


if __name__ == '__main__':
    clr_dict = {l.name: l.color for l in self_server_label}

    for name, color in clr_dict.items():
        print (name + " : " + str(color))

    label_clr = {l.categoryId: l.color for l in self_server_label}

    cur_path = os.path.realpath(__file__)
    cur_dir = os.path.dirname(cur_path)
    clr_image = os.path.join(cur_dir, 'self_clr.png')
    width = 256
    height = 1
    image = Image.new("RGBA", (width, height), (0, 0, 0))
    image_data = image.load()

    for i, color in label_clr.items():
        image_data[i, 0] = color

    image.save(clr_image)