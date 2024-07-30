import unittest , os , sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from armoInclude import * 
from armoTrain import *

## global test models
def simplTestModel() -> nn.Module:
    oModel = nn.Sequential(           
        nn.Identity(),                 # [256, 1, 500]
        nn.Flatten(),                  # Flatten to [256, 500]
        nn.Linear(in_features=500, out_features=1),  # [256, 1]
        nn.Flatten(start_dim = 0)      # Flatten to [256]
    )
    return oModel



class TestMathOperations(unittest.TestCase):

    def test_basicModel_train(self):
        """ full train cycle test"""

        ## generate data
        def __GenHarmonicData( numSignals: int, numSamples: int, samplingFreq: float, maxFreq: float, σ: float ) -> Tuple[torch.Tensor, torch.Tensor]:
            π = np.pi #<! Constant Pi
            vT   = torch.linspace(0, numSamples - 1, numSamples) / samplingFreq #<! Time samples
            vF   = maxFreq * torch.rand(numSignals)                             #<! Frequency per signal
            vPhi = 2 * π * torch.rand(numSignals)                               #<! Phase per signal
            # x_i(t) = sin(2π f_i t + φ_i) + n_i(t)
            mX = torch.sin(2 * π * vF[:, None] @ vT[None, :] + vPhi[:, None])       ## @ is matrix multiplication in PyTorch
            mX = mX + σ * torch.randn(mX.shape) #<! Add noise
            return mX, vF
        
        ## config params
        numSignalsTrain = 1_000
        numSignalsVal   = 100
        numSignalsTest  = 100
        numSamples      = 500 #<! Samples in Signal
        samplingFreq    = 100.0 #<! [Hz]
        maxFreq         = 10.0  #<! [Hz]
        σ               = 0.1 #<! Noise Std
        
        ## run params
        batchSize   = 256
        numWork     = 2 #<! Number of workers
        nEpochs     = 2

        ## generate data
        mXTrain, vYTrain    = __GenHarmonicData(numSignalsTrain, numSamples, samplingFreq, maxFreq, σ) #<! Train Data
        mXVal, vYVal        = __GenHarmonicData(numSignalsVal, numSamples, samplingFreq, maxFreq, σ)   #<! Validation Data
        mXTest, vYTest      = __GenHarmonicData(numSignalsTest, numSamples, samplingFreq, maxFreq, σ)  #<! Test Data
        ## create datasets
        dsTrain = torch.utils.data.TensorDataset(mXTrain.view(numSignalsTrain, 1, -1), vYTrain) 
        dsVal   = torch.utils.data.TensorDataset(mXVal.view(numSignalsVal, 1, -1), vYVal)
        dsTest  = torch.utils.data.TensorDataset(mXTest.view(numSignalsTest, 1, -1), vYTest)
        ## create dataloaders
        dlTrain  = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = numWork, persistent_workers = True)
        dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWork, persistent_workers = True)
        dlTest   = torch.utils.data.DataLoader(dsTest, shuffle = False, batch_size = 2 * batchSize, num_workers = numWork, persistent_workers = True)

        ## model
        oModel = simplTestModel()
        myTrain = basicModel(model=oModel, debug=True)
        ## model training
        hL = nn.MSELoss()
        hS = R2Score(num_outputs = 1)
        oOpt = torch.optim.AdamW(oModel.parameters(), lr = 1e-4, betas = (0.9, 0.99), weight_decay = 1e-5)
        oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 5e-4, total_steps = nEpochs)

        ## check
        myTrain.verify( dlTrain, dlVal, oOpt, nEpochs , batchSize , hL, hS, oSch = oSch)
        ## train
        trained = myTrain.TrainModel( dlTrain, dlVal, oOpt, nEpochs, hL, hS, oSch = oSch)
        
        ## eval on test data
        avg_test_loss = myTrain.evaluate_test_data(testData = dlTest ,loss_fn=hL)
        ## inference
        test_x_data = mXTest.view(numSignalsTest, 1, -1)
        eval_result = myTrain.inference(test_x_data)

        ## asserts
        self.assertIsNotNone(trained)
        self.assertIsNotNone(avg_test_loss)
        self.assertIsNotNone(eval_result)

        



if __name__ == '__main__':
    unittest.main()
