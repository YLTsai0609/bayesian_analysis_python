import imageio
import os
# TODO
# 維持原本gif順序，而非自訂順序，solution，filename最前面給index
root_path = '../gif/'
images = []
for root, dirs, f_list in os.walk('../gif/'):
    for f in f_list:
        f_path = root_path + f
        print('processing images...',f_path, end=" ")
        images.append(imageio.imread(f_path))

kargs = { 'duration': 1 }
# imageio.mimsave(exportname, frames, 'GIF', **kargs)
imageio.mimsave('beta.gif', images, 'GIF', **kargs)
print('Done!')
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images)
