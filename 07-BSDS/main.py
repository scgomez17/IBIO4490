#!/usr/bin/ipython3

#import lybraries
from segment_BSDS500 import segmentByClustering # Change this line if your function has a different name
from save_cells import save_cells

K=[2,10, 25, 36, 48, 55]
#save_cells('train',K)
#save_cells('val',K)
save_cells('test',K)
