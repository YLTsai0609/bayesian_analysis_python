import imageio
import os
import re
# 魔鬼藏在細節中
# 要讓檔案好好排序，不能光用index, 最好用0001, 0002, 0032 這種方式
# 才能避免 10號排在2號前面，因為按照文字來說會先排到10的"1"，要特別取出index為number又需要正則表達式
# 0001, 0002, ... 0049, 這種做法簡單有效省時間
root_path = '../gif/'
images = []

for root, dirs, f_list in os.walk('../gif/'):
    for f in sorted(f_list):
        f_path = root_path + f
        print('processing images...',f_path, end="\n")
        images.append(imageio.imread(f_path))

kargs = { 'duration': 1 }
# imageio.mimsave(exportname, frames, 'GIF', **kargs)
imageio.mimsave('beta.gif', images, 'GIF', **kargs)
print('Done!')
