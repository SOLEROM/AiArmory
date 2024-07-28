import unittest , os , sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from armoInclude import * 
from armoTrain import *


def simplTestModel( ) -> nn.Module:
        oModel = nn.Sequential(
            nn.Identity(),
            nn.Conv1d(in_channels = 1,   out_channels = 32,  kernel_size = 11), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
            nn.Conv1d(in_channels = 32,  out_channels = 64,  kernel_size = 11), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
            nn.Conv1d(in_channels = 64,  out_channels = 128, kernel_size = 11), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 11), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size = 1),
            nn.Flatten          (),
            nn.Linear           (in_features = 256, out_features = 1),
            nn.Flatten          (start_dim = 0),
            )
        return oModel


# class simplTestModel(nn.Module):
#     def __init__(self):
#         super(simplTestModel, self).__init__()
#         self.fc = nn.Linear(256, 1)  # Example model

#     def forward(self, x):
#         return self.fc(x)


class TestMathOperations(unittest.TestCase):
    

    def test_basicModel_train(self):

        def __GenHarmonicData( numSignals: int, numSamples: int, samplingFreq: float, maxFreq: float, σ: float ) -> Tuple[torch.Tensor, torch.Tensor]:
            π = np.pi #<! Constant Pi
            vT   = torch.linspace(0, numSamples - 1, numSamples) / samplingFreq #<! Time samples
            vF   = maxFreq * torch.rand(numSignals)                             #<! Frequency per signal
            vPhi = 2 * π * torch.rand(numSignals)                               #<! Phase per signal
            # x_i(t) = sin(2π f_i t + φ_i) + n_i(t)
            mX = torch.sin(2 * π * vF[:, None] @ vT[None, :] + vPhi[:, None])       ## @ is matrix multiplication in PyTorch
            mX = mX + σ * torch.randn(mX.shape) #<! Add noise
            return mX, vF
        
        numSignalsTrain = 1_000
        numSignalsVal   = 100
        numSignalsTest  = 100
        numSamples      =  500 #<! Samples in Signal
        samplingFreq    = 100.0 #<! [Hz]
        maxFreq         = 10.0  #<! [Hz]
        σ               = 0.1 #<! Noise Std

        mXTrain, vYTrain    = __GenHarmonicData(numSignalsTrain, numSamples, samplingFreq, maxFreq, σ) #<! Train Data
        mXVal, vYVal        = __GenHarmonicData(numSignalsVal, numSamples, samplingFreq, maxFreq, σ)   #<! Validation Data
        mXTest, vYTest      = __GenHarmonicData(numSignalsTest, numSamples, samplingFreq, maxFreq, σ)  #<! Test Data

        dsTrain = torch.utils.data.TensorDataset(mXTrain.view(numSignalsTrain, 1, -1), vYTrain) 
        dsVal   = torch.utils.data.TensorDataset(mXVal.view(numSignalsVal, 1, -1), vYVal)
        dsTest  = torch.utils.data.TensorDataset(mXTest.view(numSignalsTest, 1, -1), vYTest)

        batchSize   = 256
        numWork     = 2 #<! Number of workers
        nEpochs     = 2

        dlTrain  = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = numWork, persistent_workers = True)
        dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWork, persistent_workers = True)
        dlTest   = torch.utils.data.DataLoader(dsTest, shuffle = False, batch_size = 2 * batchSize, num_workers = numWork, persistent_workers = True)

        oModel = simplTestModel()

        hL = nn.MSELoss()
        hS = R2Score(num_outputs = 1)
        oOpt = torch.optim.AdamW(oModel.parameters(), lr = 1e-4, betas = (0.9, 0.99), weight_decay = 1e-5)
        oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 5e-4, total_steps = nEpochs)

        myTrain = basicModel()
        trained = myTrain.TrainModel(oModel, dlTrain, dlVal, oOpt, nEpochs, hL, hS, oSch = oSch)

        avg_test_loss = myTrain.evaluate_test_data(testData = dlTest ,loss_fn=hL)

        test_x_data = mXTest.view(numSignalsTest, 1, -1)
        eval_result = myTrain.inference(test_x_data)

        self.assertIsNotNone(trained)
        self.assertIsNotNone(avg_test_loss)
        self.assertIsNotNone(eval_result)



if __name__ == '__main__':
    unittest.main()
