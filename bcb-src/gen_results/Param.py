class Param(object):

    def __init__(self):
        self.dmin = 15
        self.dmax = 400
        self.dratio = 2
        self.d_erase = 0.50
        self.xy_erase = 0.50
        self.d_tol = 0.50
        self.xy_tol = 20
        self.stage = 0
        self.T = 121 * (7+2) + 162
        self.tt = 121 * (7+2) + 162
        self.sz_trainset = 2
        self.acceptance = False
        self.go_train = False
        self.sz_testset = 2
        self.go_test = False
        self.miu = 0.55
        self.go_classf = False
        self.n_images = 0
        self.thresh_overlay = 0.25 # the maximum shared portion between a negative sample and positive samples
        self.thresh_std = 5 # minimum standart desviation for false example
