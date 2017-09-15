import chainer
import chainer.links as L
import chainer.functions as F

class MTNNet(chainer.Chain):
    def __init__(self, n_out=20, ch=64, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(MTNNet, self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(3, ch//16, ksize=3, stride=2, pad=1, initialW=w)
            self.bn0=L.BatchNormalization(ch//16)
            
            self.c1=L.Convolution2D(ch//16, ch//8, ksize=3, stride=2, pad=1, initialW=w)
            self.bn1=L.BatchNormalization(ch//8)
        
            self.c2=L.Convolution2D(ch//8, ch//4, ksize=3, stride=2, pad=1, initialW=w)
            self.bn2=L.BatchNormalization(ch//4)
            
            self.c3=L.Convolution2D(ch//4, ch//2, ksize=3, stride=2, pad=1, initialW=w)
            self.bn3=L.BatchNormalization(ch//2)
            
            self.c4=L.Convolution2D(ch//2, ch//1, ksize=3, stride=2, pad=1, initialW=w)
            self.bn4=L.BatchNormalization(ch//1)
            
            self.l1 = L.Linear(3136, 256)
            self.l2 = L.Linear(256, n_out)
            
    def __call__(self, x):
        h = F.leaky_relu(self.bn0(self.c0(x)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.leaky_relu(self.bn4(self.c4(h)))
        
        h = F.dropout(h)
        h = self.l1(h)
        h = F.leaky_relu(h)
        
        h = F.dropout(h)
        h = self.l2(h)
        return h
    
    def loss(self, x, t):
        y = self(x)
        loss = F.softmax_cross_entropy(y,t)
        self.accuracy = F.accuracy(y,t)
        return y, loss
    