import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

for ratio in [0.3,0.5,0.7]:
    perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])
    r = ratio
    m = np.zeros((10, 10), dtype=np.float32)
    for i in xrange(10):
        m[i, i] = 1 - r
        m[i, perm[i]] = r

    plt.figure(1)
    plt.imshow(m)
    plt.title('noise ratio=%.1f' % ratio, fontsize=35)
    plt.savefig('figures/%.1f_gt.pdf' % ratio, bbox_inches='tight', dpi=100)

    m = np.load('weights/lr_m_%.1f.npy' % ratio)
    plt.figure(2)
    plt.imshow(m)
    plt.savefig('figures/%.1f_init.pdf' % ratio, bbox_inches='tight', dpi=100)

    m = np.load('weights/lr_m_%.1f_learn.npy' % ratio)
    plt.figure(3)
    plt.imshow(m)
    plt.savefig('figures/%.1f_learn.pdf' % ratio, bbox_inches='tight', dpi=100)