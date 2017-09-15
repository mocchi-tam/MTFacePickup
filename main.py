import argparse

import chainer
from chainer.dataset import convert
import chainer.functions as F

import net
import imagepp as ip

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=-1,
                        help='If positive, resume the training from snapshot')
    
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.resume < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize)
    tsm.run()
    
class MTModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        self.gpu = gpu
        self.Ncat = 20
        
        self.model = net.MTNNet(n_out=self.Ncat)
        
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()
        
        if self.flag_train:
            self.optimizer = chainer.optimizers.Adam()
            self.optimizer.setup(self.model)
        
        if self.flag_resum:
            try: 
                chainer.serializers.load_npz('./net/net.model', self.model)
                chainer.serializers.load_npz('./net/net.state', self.optimizer)
                print('successfully resume model')
            except:
                print('ERROR: cannot resume model')
        
        # prepare dataset
        filenames_train = ['./img/asai',
                           './img/harima',
                           './img/homma',
                           './img/inagaki',
                           './img/kurosu',
                           './img/maeda',
                           './img/michieda',
                           './img/muto',
                           './img/nagatomo',
                           './img/noguchi',
                           './img/sato',
                           './img/shoji',
                           './img/suzuki',
                           './img/taguchi',
                           './img/taya',
                           './img/umemoto',
                           './img/yamane',
                           './img/yamauchi',
                           './img/yasuda',
                           './img/none'
                           ]
        
        filenames_test = ['./img/asai',
                           './img/harima',
                           './img/homma',
                           './img/inagaki',
                           './img/kurosu',
                           './img/maeda',
                           './img/michieda',
                           './img/muto',
                           './img/nagatomo',
                           './img/noguchi',
                           './img/sato',
                           './img/shoji',
                           './img/suzuki',
                           './img/taguchi',
                           './img/taya',
                           './img/umemoto',
                           './img/yamane',
                           './img/yamauchi',
                           './img/yasuda',
                           './img/none'
                           ]
        
        imp = ip.ImagePP(self.gpu)
        train, _ = imp.makedataset(filenames_train, self.Ncat, train=True)
        test, self.fnames = imp.makedataset(filenames_test, self.Ncat, train=False)
        
        self.N_train = len(train)
        self.N_test = len(test)
        
        self.train_iter = chainer.iterators.SerialIterator(train, batchsize,
                                                           repeat=True, shuffle=True)
        self.test_iter = chainer.iterators.SerialIterator(test, 1,
                                                          repeat=False, shuffle=False)
        
    def run(self):
        sum_accuracy = 0
        sum_loss = 0
        
        while self.train_iter.epoch < self.n_epoch:
            # train phase
            batch = self.train_iter.next()
            if self.flag_train:
                
                # step by step update
                x, t = convert.concat_examples(batch, self.gpu)
                
                self.model.cleargrads()
                _, loss = self.model.loss(x, t)
                loss.backward()
                self.optimizer.update()
                
                sum_loss += float(loss.data) * len(t)
                sum_accuracy += float(self.model.accuracy.data) * len(t)
            
            # test phase
            if self.train_iter.is_new_epoch:
                epc = self.train_iter.epoch
                tr_acc = sum_accuracy / self.N_train
                print('epoch: ', epc)
                print('train mean loss: {}, accuracy: {}'.format(
                        sum_loss / self.N_train, tr_acc))
                
                sum_accuracy = 0
                sum_loss = 0
                
                for n_test in range(self.N_test):
                    batch_test = self.test_iter.next()
                    x, t = convert.concat_examples(batch_test, self.gpu)
                    
                    with chainer.using_config('train', False), chainer.no_backprop_mode():
                        f_t, loss = self.model.loss(x, t)
                        
                    sum_loss += float(loss.data)
                    sum_accuracy += float(self.model.accuracy.data)
                    
                    s = F.argmax(f_t).data
                    
                    print('{},{},{:.3f}'.format(self.fnames[n_test],s,int(t)))
                    
                self.test_iter.reset()
                te_acc = sum_accuracy / self.N_test
                print('test mean loss: {}, accuracy: {}'.format(
                        sum_loss / self.N_test, te_acc))
                
                sum_accuracy = 0
                sum_loss = 0
                      
        try:
            chainer.serializers.save_npz('./net/net.model', self.model)
            chainer.serializers.save_npz('./net/net.state', self.optimizer)
            print('Successfully saved model')
        except:
            print('ERROR: saving model ignored')
        
if __name__ == '__main__':
    main()