# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:21:51 2018

@author: Michael
All code is based on: https://github.com/yunjey/pytorch-tutorial
"""

import utilities

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from pathlib import Path
import pathlib

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42) # random seed to ensure weights are initialized to same vals.


# Convolutional neural network (two convolutional layers)
class ConvNet( nn.Module ):
    def __init__( self ):
        super(ConvNet, self).__init__()
        
        window_size_l1 = 1
        window_size_l2 = 7
        window_size_l3 = 15
        
        n_filters_l1   = 50
        n_filters_l2   = 11
        n_filters_l3   = 3
        
        kernel_size_l1 = ( window_size_l1, 1 )
        kernel_size_l2 = ( window_size_l2, 1 )
        kernel_size_l3 = ( window_size_l3, 1 )
        
        pad_l1         = self._get_padding( 'SAME', kernel_size_l1 )
        pad_l2         = self._get_padding( 'SAME', kernel_size_l2 )
        pad_l3         = self._get_padding( 'SAME', kernel_size_l3 )
        self.dropout   = nn.Dropout( p = 0.3 )
        self.cnn1    = nn.Conv2d( in_channels=2,          out_channels=n_filters_l1, 
                                 kernel_size=kernel_size_l1, padding=pad_l1 )
        self.cnn2    = nn.Conv2d( in_channels=n_filters_l1, out_channels=n_filters_l2, 
                                 kernel_size=kernel_size_l2, padding=pad_l2 )
        self.cnn3    = nn.Conv2d( in_channels=n_filters_l2, out_channels=n_filters_l3,
                                 kernel_size=kernel_size_l3, padding=pad_l3 )
        
        
        '''
        window_size_l1 = 5
        window_size_l2 = 11
        window_size_l3 = 17

        n_filters_l1   = 68
        n_filters_l2   = 32
        n_filters_l3   = 16
        kernel_size_l1 = ( window_size_l1, 1 )
        kernel_size_l2 = ( window_size_l2, 1 )
        kernel_size_l3 = ( window_size_l3, 1 )
        pad_l1         = self._get_padding( 'SAME', kernel_size_l1 )
        pad_l2         = self._get_padding( 'SAME', kernel_size_l2 )
        pad_l3         = self._get_padding( 'SAME', kernel_size_l3 )
        self.dropout   = nn.Dropout( p = 0.3 )
        self.layer1    = nn.Conv2d( in_channels=100, out_channels=n_filters_l1, kernel_size=kernel_size_l1, padding=pad_l1 )
        self.layer2    = nn.Conv2d( in_channels=n_filters_l1, out_channels=n_filters_l2, kernel_size=kernel_size_l2, padding=pad_l2 )
        self.layer3    = nn.Conv2d( in_channels=n_filters_l2, out_channels=n_filters_l3, kernel_size=kernel_size_l3, padding=pad_l3 )
        self.fc1       = nn.Linear( n_filters_l3, 3 )
        
        self.cnn_1     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(3,1),padding=self._get_padding( 'SAME',(3,1) ))
        self.cnn_2     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(5,1),padding=self._get_padding( 'SAME',(5,1) ))
        self.cnn_3     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(7,1),padding=self._get_padding( 'SAME',(7,1) ))
        self.cnn_4     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(9,1),padding=self._get_padding( 'SAME',(9,1) ))
        self.cnn_5     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(11,1),padding=self._get_padding( 'SAME',(11,1) ))
        self.cnn_6     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(13,1),padding=self._get_padding( 'SAME',(13,1) ))
        self.cnn_7     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(15,1),padding=self._get_padding( 'SAME',(15,1) ))
        self.cnn_8     = nn.Conv2d( in_channels=100, out_channels=10, kernel_size=(17,1),padding=self._get_padding( 'SAME',(17,1) ))
        
        self.hidden_dim = 10
        self.hidden       = self.init_hidden() # initialize hidden state
        self.lstm         = nn.LSTM( 80, self.hidden_dim, bidirectional=True )  # Input dim is 3, output dim is 3
        self.dense        = nn.Linear( in_features=2*self.hidden_dim, out_features=3 ) # reduce to 3 predicted classes
        
        
        self.fully1    = nn.Linear( in_features=500, out_features=100 )
        self.fully2    = nn.Linear( in_features=100, out_features=10 )
        self.fully3    = nn.Linear( in_features=10, out_features=3 )
        
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(10)
        #self.softmax   = F.log_softmax(dim=1) # Channel dimension contains class probabilities
        #self.fc = nn.Linear( n_filters, DataSplitter.NCLASSES )
        '''
    # https://github.com/pytorch/pytorch/issues/3867
    def _get_padding( self, padding_type, kernel_size ):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return tuple((k - 1) // 2 for k in kernel_size)
        return tuple(0 for _ in kernel_size)
    
    def init_hidden(self):
        # See: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 64, self.hidden_dim),
                torch.zeros(2, 64, self.hidden_dim))
    '''
    def forward( self, x):
        
        out_1 = F.relu( self.cnn_1(x))
        out_2 = F.relu( self.cnn_2(x))
        out_3 = F.relu( self.cnn_3(x))
        out_4 = F.relu( self.cnn_4(x))
        out_5 = F.relu( self.cnn_5(x))
        out_6 = F.relu( self.cnn_6(x))
        out_7 = F.relu( self.cnn_7(x))
        out_8 = F.relu( self.cnn_8(x))
        out = torch.cat( [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8], dim=1 )


        #out = out.squeeze().permute(1,0)
        #out = out.unsqueeze(1) # add (empty) batch_size dim to fit lstm shape-expectation: seq x batch x features
        out = out.squeeze().permute(2,0,1)
        
        out, self.hidden = self.lstm( out , self.hidden )
        #out = out.permute(0,3,2,1) # swap feature dimension to last axis as dense layers expect N,...,C
        out = F.log_softmax( self.dense(out), dim=-1 )
        #out = out.permute(0,3,2,1) # swap feature dimension (now: predictions) back to second axis
        
        #out = out.permute(1,2,0).unsqueeze(-1) # swap & add dims to fit batch_size x C x Sequence x 1
        out = out.permute(1,2,0).unsqueeze(-1)
        return out
    '''
    def forward( self, x):
        out = self.dropout( F.relu(        self.cnn1( x )) )
        out = self.dropout( F.relu(        self.cnn2(out)) )
        out =               F.log_softmax( self.cnn3(out), dim=1 )
        #out = out.permute(0,3,2,1)

        #out = F.log_softmax( self.fc1(out), dim=1 )
        #out = out.permute(0,3,2,1) # swap feature dimension (now: predictions) back to second axis
        return out
    


class CustomDataset( torch.utils.data.Dataset ):
    
    LEN_FEATURE_VEC = 100
    
    def __init__( self, samples, max_len=None ):
        # 1. Initialize file paths or a list of file names.
        self.max_len      = max_len # if no max_len is provided, assume that only a single sequence should be processed at a time
        self.to_tensor    = transforms.ToTensor() # transform numpy (CPU) to tensor (GPU)
        self.inputs, self.targets = zip(*[ (inputs, targets)  
                                        for _, (inputs, targets) in samples.items() ])

        self.data_len     = len( self.inputs ) # number of samples in the set
        
    
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        X      = self.inputs[  index ]
        target = self.targets[ index ]
        
        Y       = target[0,:] # 0:1 keeps singleton dimension of target array
        mask    = target[1,:]
        
        # 1.) Read in sequence encoded by perotVec representations
        # 2.) Create sliding window view on np.memmap with window size=5
        # 3.) Transpose
        # 4.) Pytorch convention: (N_samples x Channels x Width x Height) with N_samples == batchsize which is added during read in
        if type(X) is pathlib.PosixPath: # read in numpy memmap if necessary
            X = self.read_in_numpy( X, Y )
        
        if self.max_len: # if a maximum length (longest seq. in the set) was provided
            n_missing = self.max_len - Y.size # get the number of residues which have to be padded to achieve same size for all proteins in the set
            Y    = np.pad( Y, pad_width=(0, n_missing), mode='constant', constant_values=0  )
            X    = np.pad( X, pad_width=((0,n_missing),(0,0)), mode='constant', constant_values=0. )
            mask = np.pad( mask, pad_width=(0,n_missing), mode='constant', constant_values=0)

            
        #X = self.get_sliding_window_view( X )
        X = X.T
        X    = np.expand_dims( X,    axis=2 )
        Y    = np.expand_dims( Y,    axis=1 ) # PyTorch expects target to be 3D, so add an empty dim.
        mask = np.expand_dims( mask, axis=1 )
        
        input_tensor  = torch.from_numpy( X )
        output_tensor = torch.from_numpy( Y )
        mask_tensor   = torch.from_numpy( mask ).float() # cast to float

        return ( input_tensor, output_tensor, mask_tensor )
    
    def __len__( self ):
        # You should change 0 to the total size of your dataset.
        return self.data_len
    
    def read_in_numpy( self, path, arr ):
        '''
        Reads in numpy array
        '''
        length = arr.shape[0] # shape: ( batch x sequence x classes )
        shape  = ( length, self.LEN_FEATURE_VEC ) # shape of input memmap ( L x size of vector representation )
        protVec_input  = np.memmap( path, dtype=np.float32, mode='r', shape = shape)
        return protVec_input
    
    # https://gist.github.com/Fnjn/b061b28c05b5b0e768c60964d2cafa8d
    def get_sliding_window_view( self, arr, window_size=5 ):
        '''
            Input: 
                arr = numpy array of shape ( length, n_features ) with L being
                        the length of the protein and n_features being the
                        length of the vector representation created by ProtVec.
                window_size = gives the size of the sliding window.
            Output:
                view = view on the original numpy array. Now the shape is 
                        ( length, n_features_in_window ) with n_features_in_window
                        being the length of the vector representation times the
                        window_size.
                        
            Return a view on the original array. (comparable to pointers in C)
            Each row in the view contains now also the information from neighbouring
            residues. This allows to use the view as an input to scikit learn classifiers
            which require the input shape of ( samples, n_features ).
            Zero-Padding is performed, s.t. the length of the array stays the same.
        '''
        
        length, n_features = arr.shape
        # get numbers of rows needed for proper zero-padding. Round to next int inplace
        pad_size           = window_size // 2 
        
        # pad sample axis (rows) with zeros; do not pad feature dimension (columns)!
        padded_arr = np.pad(arr, pad_width=((pad_size, pad_size), (0, 0)), mode='constant', constant_values=0. )
        
        # output shape still has (length) rows (aka samples) but 
        # has now (window_size*n_features) columns (aka features)
        view_shape   = np.array( (length, window_size*n_features ), np.int ) 
        view_strides = arr.strides # keep size of strides
        pad_strides  = padded_arr.strides # do a simple check to avoid errors due to
        assert view_strides == pad_strides, 'Padding changed strides!'

        view = np.lib.stride_tricks.as_strided( 
                padded_arr, shape=view_shape, strides=view_strides, writeable=False)
        return view


class DataSplitter():
    '''
        Takes path to directory holding input data and path to directory holding 
        targets.
        Returns sets of tuples { (input_paths, target_memmap}  for train, test & validation samples.
        target_memmap itself consists of channel0: targets and channel1: mask
    '''
    
    # class enumeration has to start with 1, ... see: https://github.com/torch/nn/issues/471
    DSSP3_MAPPING = { 'C' : 0,
                      'H' : 1,
                      'E' : 2,
                      'Y' : 2, # map ambigious residues also to E, mask out later
                      'X' : 2  # map missing residues also to E, mask out later
                      }
    
    DSSP3_NAN = { 'X', 'Y' } # characters which do not reflect an actual secStruct class 
    
    # get numpy array of unique class labels
    CLASSES = np.unique( np.fromiter([0,1,2], dtype=np.int ))
    NCLASSES= len(CLASSES)
    
    def __init__( self, input_dir, target_dir, map_path, get_statistics=True ):
        
        self.get_statistics = get_statistics
        if self.get_statistics: # if statistics on the data is needed -> Count
            from collections import Counter
            self.counter_all    = Counter()
            
        self.max_len     = 0 # container for saving length of longest sequence in the data set
        self.sifts = utilities.get_sifts_mapping( map_path )

        self.targets     = self.get_output_data( target_dir )
        if input_dir.name.endswith('.txt'):
            self.input_paths = self.get_codon_inputs( input_dir, map_path )
        else:
            self.input_paths = self.get_input_paths( input_dir )
        

        if self.get_statistics:
            print('Label distribution for all annotations: {}'.format( self.counter_all ))
            
    def get_max_length( self ): # return length of longest sequence in the set
        return self.max_len
        
    def get_output_data( self, output_dir ):
        '''
            Read in fasta-formatted file containing dssp3 annotations for each residue
            in the sequence instead of an amino acid.
            Returns Numpy array containing an index for each nominal class (C,H,E).
            Also returns a mask indicating whether a residue was actually observed
            in the structure (1) or not (0). Latter cases will be masked out later.
        '''

        def _read_in_dssp( dssp_path ):
            dssp_data = ''
            with open( dssp_path, 'rU' ) as dssp_f:
                next(dssp_f) # skip header
                for line in dssp_f:
                    dssp_data = ''.join( [dssp_data, line] ) # append string
            return ''.join( dssp_data.split() ) # remove any new line characters etc
        
        output_data = dict()
        for dssp_path in output_dir.glob('**/*.3.consensus.dssp'):

            pdb_id    = dssp_path.name.replace( '.3.consensus.dssp', '' )
            dssp_str  = _read_in_dssp( dssp_path ) # read in dssp file
                
            if self.get_statistics:
                self.counter_all.update( dssp_str ) # count all labels (including X,Y)
                    
            if len(dssp_str) > self.max_len: # save length of longest sequence in the set
                self.max_len = len(dssp_str)
                
            dssp_num  = [ self.DSSP3_MAPPING[dssp_state]  for dssp_state in dssp_str ]
                
            # create list of residues which have to be masked out during training
            mask      = [ 0 if dssp_state in self.DSSP3_NAN else 1 
                                 for dssp_state in dssp_str ]
                
            # transform to numpy array
            dssp_data = np.asarray( dssp_num, dtype=np.int )
            mask_data = np.asarray( mask,     dtype=np.int )
                
            # stack every target/mask combination, s.t. target=index0, mask=index1
            output_data[ pdb_id ] = np.vstack( (dssp_data, mask_data) )
                
        return output_data
    
    def get_codon_inputs( self, input_path, map_path ):
        uniprot2pdb = dict()
        with open( map_path, 'rU' ) as map_f:
            for line in map_f:
                pdb_id     = line.split()[0]
                uniprot_id = line.split()[2]
                
                uniprot2pdb[ uniprot_id ] = pdb_id
        
        input_dict = dict()
        with open( input_path, 'rU' ) as codon_f:
            next( codon_f )
            for line in codon_f:
                if '?' in line: 
                    continue
                uniprot_id, rscu, unpaired_prob = line.split()
                pdb_id        = uniprot2pdb[ uniprot_id ]
                sp_beg = 9999
                sp_end = 0
                try:
                    mapping = self.sifts[ pdb_id ]
                except KeyError:
                    continue
                
                for chain, entry in mapping.items():
                    sp_beg = min( sp_beg, entry['SP_BEG'] )
                    sp_end = max( sp_end, entry['SP_END'])

                rscu          = np.fromstring( rscu,          sep=',', dtype=np.float32 )[sp_beg-1:sp_end]
                unpaired_prob = np.fromstring( unpaired_prob, sep=',', dtype=np.float32 )[sp_beg-1:sp_end]
                input_dict[ pdb_id ] = np.vstack( (rscu, unpaired_prob) ).T

        return input_dict
    
    def get_input_paths( self, input_dir ):
        input_path_dict = dict() # container for input paths.
        for input_path in input_dir.glob('**/*.memmap'):
                
            pdb_id = input_path.name.replace('.memmap','')
            if not pdb_id in self.targets: # skip proteins with no target
                continue

            input_path_dict[ pdb_id ] = input_path
        return input_path_dict
    
    def split_data( self ):    
        train = dict()
        test  = dict()
        val   = dict()
        np.random.seed(42) # set a seed to be able to reproduce the splits
        
        n_missing = 0
        #TODO: Remove sorted list. But this keeps it consistent while debugging
        for pdb_id in sorted( list( self.input_paths.keys())):
            rnd = np.random.rand()
            inputs  = self.input_paths[  pdb_id ]
            try:
                targets = self.targets[ pdb_id ]
            except KeyError:
                n_missing += 1
                continue
            
            if not type( inputs ) is pathlib.PosixPath:
                try:
                    assert targets.shape[1] == inputs.shape[0]
                except AssertionError:
                    res_beg = 9999
                    res_end = 0
                    mapping = self.sifts[ pdb_id ]
                    for chain, entry in mapping.items():
                        res_beg = min( res_beg, entry['RES_BEG'] )
                        res_end = max( res_end, entry['RES_END'] )
                    targets = targets[:, res_beg-1:res_end]

                    try:
                        assert targets.shape[1] == inputs.shape[0], print( 
                            pdb_id, targets.shape, inputs.shape )
                    except AssertionError:
                        print(pdb_id, targets.shape, inputs.shape)
                        continue
                    
            if rnd > 0.9:
                val[   pdb_id ] = ( inputs, targets )
            elif rnd > 0.75:
                test[  pdb_id ] = ( inputs, targets )
            else:
                if len(train) > 0: continue
                train[ pdb_id ] = ( inputs, targets )
                
        print('Number of samples without target: {}'.format(n_missing))
        print('Size of training set: {}'.format(   len(train) ))
        print('Size of test set: {}'.format(       len(test)  ))
        print('Size of validation set: {}'.format( len(val)   ))
        
        return train, test, val
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def training( model, optimizer, criterion, train_loader ):
    
    model.train() # ensure model is in training mode (dropout, batch norm, ...)
    for i, ( inputs, targets, mask ) in enumerate(train_loader):

        optimizer.zero_grad() # zeroes the gradient buffers of all parameters

        X      = inputs.to(device)
        Y_true = targets.to(device)
        mask   = mask.to(device)

        # Forward pass
        Y_raw = model( X )
        loss   = criterion( Y_raw, Y_true )
        
        loss *= mask # maks out loss for unresolved/ambigious residues
        loss  = loss.sum() / mask.sum() # take average while removing masked out samples
        
        loss.backward()
        optimizer.step()


def testing( model, data_loaders, criterion, epoch, eval_summary ):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        running_loss   = ''
        for test_set_name, test_loader in data_loaders.items(): # for Train & Test
            
            # reset performance statistics & make it work on GPU
            loss_avg     = torch.zeros( 1,  dtype=torch.float, device=device, requires_grad=False)
            Y_true_all   = np.zeros( 0,     dtype=np.int )
            Y_pred_all   = np.zeros( 0,     dtype=np.int )
            conf_mat     = np.zeros( (3,3), dtype=np.int )
            
            for inputs, targets, mask in test_loader:
                X      = inputs.to(device)
                Y_true = targets.to(device)
                mask   = mask.to(device)
            
                Y_raw  = model( X )
                loss_avg  += ((criterion( Y_raw, Y_true ) * mask).sum()) / mask.sum()
            
                _, Y_pred = torch.max( Y_raw.data, dim=1 ) # returns index of output predicted with highest probability
                Y_pred = Y_pred[ mask==1 ]
                Y_true = Y_true[ mask==1 ] # remove ambigious or unresolved residues
                
                Y_true = Y_true.view(-1).long().cpu().numpy()
                Y_pred = Y_pred.view(-1).long().cpu().numpy()
                
                np.add.at( conf_mat, ( Y_true, Y_pred), 1 )
                
                Y_true_all = np.append( Y_true_all, Y_true )
                Y_pred_all = np.append( Y_pred_all, Y_pred)

            loss_avg = ( loss_avg/len(test_loader) ).cpu().numpy()[0]
            running_loss += '{0}: {1:.3f} '.format( test_set_name, loss_avg )

            utilities.add_progress( eval_summary, test_set_name, loss_avg, epoch,
                                       conf_mat, Y_true_all, Y_pred_all )
    print( running_loss )
def main():
    # TODO: Use MSA ProtVec as Input
    # TODO: ADD standard errors

    root_dir   = Path('/home/mheinzinger/contact_prediction_v2/')
    #input_dir  = root_dir / 'inputs'  / 'protVec_original_seq'
    input_dir  = root_dir / 'inputs' / 'uniprot_rscu_pp2.txt'
    output_dir = root_dir / 'targets' / 'dssp'
    log_dir    = root_dir / 'sec_struct_jan' / 'log'
    map_path   = root_dir / 'sec_struct_jan' / 'ss_pdb_ids.txt'
    #log_path   = log_dir  / 'protvec_cnn-50x1-11x7-3x15_lr1e-3_n-100_batch32'
    log_path   = log_dir  / 'testCNN_codonData_1train'
    

    utilities.create_logdir( log_path )
    
    # create tuples containing ( path_2_input_sample, path_2_corresponding_target )
    # and split in train, test, val
    data_splitter    = DataSplitter( input_dir, output_dir, map_path ) # read in data
    train, test, val = data_splitter.split_data() # split data into 3 sets
    max_length       = data_splitter.get_max_length() # get length of longest sequence in the set
    
    train_set = CustomDataset( train, max_length )
    test_set  = CustomDataset( test,  max_length )

    
    
    data_loaders = dict()
    data_loaders['Train'] = torch.utils.data.DataLoader( dataset=train_set,
                                                batch_size=32, 
                                                shuffle=True,
                                                drop_last=False
                                                #num_workers=2
                                                )
    data_loaders['Test'] = torch.utils.data.DataLoader( dataset=test_set,
                                                batch_size=100, 
                                                shuffle=False,
                                                drop_last=False
                                                )
    
    eval_summary = dict()
    # Hyper parameters
    num_epochs    = 100
    learning_rate = 1e-3
    
    model = ConvNet().to(device)
    print('Number of free parameters: {}'.format( count_parameters(model) ))
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss( reduce=False )
    optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate, amsgrad=True )

    # Train the model
    for epoch in range(num_epochs):
        training( model, optimizer, criterion, data_loaders['Train'] )
        testing(  model, data_loaders, criterion, epoch, eval_summary )
        utilities.save_performance( log_path, eval_summary )
        utilities.plot_learning( log_path, eval_summary )
            
    utilities.plot_learning( log_path, eval_summary )
    utilities.save_performance( log_path, eval_summary, final_flag=True )
    torch.save( model.state_dict(), log_path / 'model.ckpt' )
    
    
if __name__ == '__main__':
    main()