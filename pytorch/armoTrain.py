
# Python STD
import os
from enum import auto, Enum, unique
import math
import time
# OOP
from abc import ABC, abstractmethod

# Data
import numpy as np
import scipy as sp
# Machine Learning

# Deep Learning
import  torch
import  torch.nn            as nn
import  torch.nn.functional as F
from    torch.optim.optimizer import Optimizer
from    torch.optim.lr_scheduler import LRScheduler
from    torch.utils.data import DataLoader, Dataset
from    torch.utils.data import default_collate
from    torch.utils.tensorboard import SummaryWriter
import  torchvision
from    torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Course Packages
from DeepLearningBlocks import NNMode

# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union


class TBLogger():
    def __init__( self, logDir: Optional[str] = None ) -> None:

        self.oTBWriter  = SummaryWriter(log_dir = logDir)
        self.iiEpcoh    = 0
        self.iiItr      = 0
        
        pass

    def close( self ) -> None:

        self.oTBWriter.close()

class BaseClass(ABC):
    
    def __init__(self,CheckpointFile:str):
        self.model = None
        self.lTrainScore = None
        self.lTrainLoss = None
        self.lValLoss = None
        self.lValScore = None
        self.lLearnRate = None
        self.CheckpointFile = CheckpointFile

    def plotTrainResults(self):
        hF, vHa = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 5))
        vHa = np.ravel(vHa)

        hA = vHa[0]
        hA.plot(self.lTrainLoss, lw = 2, label = 'Train')
        hA.plot(self.lValLoss, lw = 2, label = 'Validation')
        hA.set_title('Binary Cross Entropy Loss')
        hA.set_xlabel('Epoch')
        hA.set_ylabel('Loss')
        hA.legend()

        hA = vHa[1]
        hA.plot(self.lTrainScore, lw = 2, label = 'Train')
        hA.plot(self.lValScore, lw = 2, label = 'Validation')
        hA.set_title('Accuracy Score')
        hA.set_xlabel('Epoch')
        hA.set_ylabel('Score')
        hA.legend()

        hA = vHa[2]
        hA.plot(self.lLearnRate, lw = 2)
        hA.set_title('Learn Rate Scheduler')
        hA.set_xlabel('Epoch')
        hA.set_ylabel('Learn Rate')


    def evaluate_test_data(self,testData, loss_fn):
        if not self.model:
            raise ValueError(f'no AI model!')

        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in testData:
                inputs, targets = batch
                outputs = self.model(inputs)
                test_loss += loss_fn(outputs, targets).item()
        avg_test_loss = test_loss / len(testData)
        print(f"Test Loss: {avg_test_loss}")
        return avg_test_loss


    def inference(self,input):
        if not self.model:
            raise ValueError(f'no AI model!')
        output = None        
        try:
            self.model.eval()
            with torch.no_grad():  
                output = self.model(input)
        except:
            print(f" armoTrain::inference  FAILED to get output")
        return output

class basicModel(BaseClass):

    def __init__(self,CheckpointFile:str=None):
        BaseClass.__init__(self,CheckpointFile)


    def RunEpoch(self, oModel: nn.Module, dlData: DataLoader, hL: Callable, hS: Callable, oOpt: Optional[Optimizer] = None, opMode: NNMode = NNMode.TRAIN ) -> Tuple[float, float]:
        """
        Runs a single Epoch (Train / Test) of a model.  
        Input:
            oModel      - PyTorch `nn.Module` object.
            dlData      - PyTorch `Dataloader` object.
            hL          - Callable for the Loss function.
            hS          - Callable for the Score function.
            oOpt        - PyTorch `Optimizer` object.
            opMode      - An `NNMode` to set the mode of operation.
        Output:
            valLoss     - Scalar of the loss.
            valScore    - Scalar of the score.
        Remarks:
        - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
        - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
            It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
        - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
            It should return a scalar `valScore` of the score.
        - The optimizer is required for training mode.
        """
        
        epochLoss   = 0.0
        epochScore  = 0.0
        numSamples  = 0
        numBatches = len(dlData)

        runDevice = next(oModel.parameters()).device #<! CPU \ GPU

        if opMode == NNMode.TRAIN:
            oModel.train(True) #<! Equivalent of `oModel.train()`
        elif opMode == NNMode.INFERENCE:
            oModel.eval() #<! Equivalent of `oModel.train(False)`
        else:
            raise ValueError(f'The `opMode` value {opMode} is not supported!')
        
        for ii, (mX, vY) in enumerate(dlData):
            # Move Data to Model's device
            mX = mX.to(runDevice) #<! Lazy
            vY = vY.to(runDevice) #<! Lazy

            batchSize = mX.shape[0]
            
            if opMode == NNMode.TRAIN:
                # Forward
                mZ      = oModel(mX) #<! Model output
                valLoss = hL(mZ, vY) #<! Loss
                
                # Backward
                oOpt.zero_grad()    #<! Set gradients to zeros
                valLoss.backward()  #<! Backward
                oOpt.step()         #<! Update parameters
            else: #<! Value of `opMode` was already validated
                with torch.no_grad():
                    # No computational graph
                    mZ      = oModel(mX) #<! Model output
                    valLoss = hL(mZ, vY) #<! Loss

            with torch.no_grad():
                # Score
                valScore = hS(mZ, vY)
                # Normalize so each sample has the same weight
                epochLoss  += batchSize * valLoss.item()
                epochScore += batchSize * valScore.item()
                numSamples += batchSize

            print(f'\r{"Train" if opMode == NNMode.TRAIN else "Val"} - Iteration: {(ii + 1):3d} / {numBatches}, loss: {valLoss:.6f}', end = '')
        
        print('', end = '\r')
                
        return epochLoss / numSamples, epochScore / numSamples

    def TrainModel(self, oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, numEpoch: int, hL: Callable, hS: Callable, *, oSch: Optional[LRScheduler] = None, oTBWriter: Optional[SummaryWriter] = None) -> Tuple[nn.Module, List, List, List, List]:
        """
        Trains a model given test and validation data loaders.  
        Input:
            oModel      - PyTorch `nn.Module` object.
            dlTrain     - PyTorch `Dataloader` object (Training).
            dlVal       - PyTorch `Dataloader` object (Validation).
            oOpt        - PyTorch `Optimizer` object.
            numEpoch    - Number of epochs to run.
            hL          - Callable for the Loss function.
            hS          - Callable for the Score function.
            oSch        - PyTorch `Scheduler` (`LRScheduler`) object.
            oTBWriter   - PyTorch `SummaryWriter` object (TensorBoard).
        Output:
            lTrainLoss     - Scalar of the loss.
            lTrainScore    - Scalar of the score.
            lValLoss    - Scalar of the score.
            lValScore    - Scalar of the score.
            lLearnRate    - Scalar of the score.
        Remarks:
        - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
        - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
            It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
        - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
            It should return a scalar `valScore` of the score.
        - The optimizer is required for training mode.
        """

        lTrainLoss  = []
        lTrainScore = []
        lValLoss    = []
        lValScore   = []
        lLearnRate  = []

        # Support R2
        bestScore = -1e9 #<! Assuming higher is better

        learnRate = oOpt.param_groups[0]['lr']

        for ii in range(numEpoch):
            startTime           = time.time()
            trainLoss, trainScr = self.RunEpoch(oModel, dlTrain, hL, hS, oOpt, opMode = NNMode.TRAIN) #<! Train
            valLoss,   valScr   = self.RunEpoch(oModel, dlVal, hL, hS, oOpt, opMode = NNMode.INFERENCE) #<! Score Validation
            if oSch is not None:
                # Adjusting the scheduler on Epoch level
                learnRate = oSch.get_last_lr()[0]
                oSch.step()
            epochTime           = time.time() - startTime

            # Aggregate Results
            lTrainLoss.append(trainLoss)
            lTrainScore.append(trainScr)
            lValLoss.append(valLoss)
            lValScore.append(valScr)
            lLearnRate.append(learnRate)

            if oTBWriter is not None:
                oTBWriter.add_scalars('Loss (Epoch)', {'Train': trainLoss, 'Validation': valLoss}, ii)
                oTBWriter.add_scalars('Score (Epoch)', {'Train': trainScr, 'Validation': valScr}, ii)
                oTBWriter.add_scalar('Learning Rate', learnRate, ii)
            
            # Display (Babysitting)
            print('Epoch '              f'{(ii + 1):4d} / ' f'{numEpoch}', end = '')
            print(' | Train Loss: '     f'{trainLoss          :6.3f}', end = '')
            print(' | Val Loss: '       f'{valLoss            :6.3f}', end = '')
            print(' | Train Score: '    f'{trainScr           :6.3f}', end = '')
            print(' | Val Score: '      f'{valScr             :6.3f}', end = '')
            print(' | Epoch Time: '     f'{epochTime          :5.2f}', end = '')

            # Save best model ("Early Stopping")
            if valScr > bestScore:
                bestScore = valScr
                try:
                    dCheckPoint = {'Model': oModel.state_dict(), 'Optimizer': oOpt.state_dict()}
                    if oSch is not None:
                        dCheckPoint['Scheduler'] = oSch.state_dict()
                    if self.CheckpointFile:
                        torch.save(dCheckPoint, 'BestModel.pt')
                        print(' | <-- Checkpoint Saved!', end = '')
                except:
                    print(' | <-- Failed!', end = '')
            print(' |')
        
        # Load best model ("Early Stopping")
        # dCheckPoint = torch.load('BestModel.pt')
        # oModel.load_state_dict(dCheckPoint['Model'])
        self.model = oModel
        self.lTrainLoss = lTrainLoss
        self.lTrainScore = lTrainScore
        self.lValLoss = lValLoss
        self.lValScore = lValScore
        self.lLearnRate = lLearnRate

        return self
    



class schModel(BaseClass):
    
    def __init__(self,CheckpointFile:str=None):
        BaseClass.__init__(self,CheckpointFile)


    def RunEpoch( oModel: nn.Module, dlData: DataLoader, hL: Callable, hS: Callable, oOpt: Optional[Optimizer] = None, oSch: Optional[LRScheduler] = None, opMode: NNMode = NNMode.TRAIN, oTBLogger: Optional[TBLogger] = None ) -> Tuple[float, float]:
        """
        Runs a single Epoch (Train / Test) of a model.  
        Supports per iteration (Batch) scheduling. 
        Input:
            oModel      - PyTorch `nn.Module` object.
            dlData      - PyTorch `Dataloader` object.
            hL          - Callable for the Loss function.
            hS          - Callable for the Score function.
            oOpt        - PyTorch `Optimizer` object.
            oSch        - PyTorch `Scheduler` (`LRScheduler`) object.
            opMode      - An `NNMode` to set the mode of operation.
            oTBLogger   - An `TBLogger` object.
        Output:
            valLoss     - Scalar of the loss.
            valScore    - Scalar of the score.
            learnRate   - Scalar of the average learning rate over the epoch.
        Remarks:
        - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
        - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
            It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
        - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
            It should return a scalar `valScore` of the score.
        - The optimizer / scheduler are required for training mode.
        """
        
        epochLoss   = 0.0
        epochScore  = 0.0
        numSamples  = 0
        epochLr     = 0.0
        numBatches = len(dlData)
        lLearnRate = []

        runDevice = next(oModel.parameters()).device #<! CPU \ GPU

        if opMode == NNMode.TRAIN:
            oModel.train(True) #<! Equivalent of `oModel.train()`
        elif opMode == NNMode.INFERENCE:
            oModel.eval() #<! Equivalent of `oModel.train(False)`
        else:
            raise ValueError(f'The `opMode` value {opMode} is not supported!')
        
        for ii, (mX, vY) in enumerate(dlData):
            # Move Data to Model's device
            mX = mX.to(runDevice) #<! Lazy
            vY = vY.to(runDevice) #<! Lazy

            batchSize = mX.shape[0]
            
            if opMode == NNMode.TRAIN:
                # Forward
                mZ      = oModel(mX) #<! Model output
                valLoss = hL(mZ, vY) #<! Loss
                
                # Backward
                oOpt.zero_grad()    #<! Set gradients to zeros
                valLoss.backward()  #<! Backward
                oOpt.step()         #<! Update parameters

                learnRate = oSch.get_last_lr()[0]
                oSch.step() #<! Update learning rate

            else: #<! Value of `opMode` was already validated
                with torch.no_grad():
                    # No computational graph
                    mZ      = oModel(mX) #<! Model output
                    valLoss = hL(mZ, vY) #<! Loss
                    
                    learnRate = 0.0

            with torch.no_grad():
                # Score
                valScore = hS(mZ, vY)
                # Normalize so each sample has the same weight
                epochLoss  += batchSize * valLoss.item()
                epochScore += batchSize * valScore.item()
                epochLr    += batchSize * learnRate
                numSamples += batchSize
                lLearnRate.append(learnRate)

                if (oTBLogger is not None) and (opMode == NNMode.TRAIN):
                    # Logging at Iteration level for training
                    oTBLogger.iiItr += 1
                    oTBLogger.oTBWriter.add_scalar('Train Loss', valLoss.item(), oTBLogger.iiItr)
                    oTBLogger.oTBWriter.add_scalar('Train Score', valScore.item(), oTBLogger.iiItr)
                    oTBLogger.oTBWriter.add_scalar('Learning Rate', learnRate, oTBLogger.iiItr)

            print(f'\r{"Train" if opMode == NNMode.TRAIN else "Val"} - Iteration: {(ii + 1):3d} / {numBatches}, loss: {valLoss:.6f}', end = '')
        
        print('', end = '\r')
            
        return epochLoss / numSamples, epochScore / numSamples, epochLr / numSamples, lLearnRate


    def TrainModelSch(self, oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, oSch: LRScheduler, numEpoch: int, hL: Callable, hS: Callable, oTBLogger: Optional[TBLogger] = None ) -> Tuple[nn.Module, List, List, List, List]:

        lTrainLoss  = []
        lTrainScore = []
        lValLoss    = []
        lValScore   = []
        lLearnRate  = []

        # Support R2
        bestScore = -1e9 #<! Assuming higher is better

        for ii in range(numEpoch):
            startTime                               = time.time()
            trainLoss, trainScr, trainLr, lLRate    = self.RunEpoch(oModel, dlTrain, hL, hS, oOpt, oSch, opMode = NNMode.TRAIN, oTBLogger = oTBLogger) #<! Train
            valLoss,   valScr, *_                   = self.RunEpoch(oModel, dlVal, hL, hS, opMode = NNMode.INFERENCE)    #<! Score Validation
            epochTime                               = time.time() - startTime

            # Aggregate Results
            lTrainLoss.append(trainLoss)
            lTrainScore.append(trainScr)
            lValLoss.append(valLoss)
            lValScore.append(valScr)
            lLearnRate.extend(lLRate)

            if oTBLogger is not None:
                oTBLogger.iiEpcoh += 1
                oTBLogger.oTBWriter.add_scalars('Loss (Epoch)', {'Train': trainLoss, 'Validation': valLoss}, ii)
                oTBLogger.oTBWriter.add_scalars('Score (Epoch)', {'Train': trainScr, 'Validation': valScr}, ii)
                oTBLogger.oTBWriter.add_scalar('Learning Rate (Epoch)', trainLr, ii)
                oTBLogger.oTBWriter.flush()
            
            # Display (Babysitting)
            print('Epoch '              f'{(ii + 1):4d} / ' f'{numEpoch}', end = '')
            print(' | Train Loss: '     f'{trainLoss          :6.3f}', end = '')
            print(' | Val Loss: '       f'{valLoss            :6.3f}', end = '')
            print(' | Train Score: '    f'{trainScr           :6.3f}', end = '')
            print(' | Val Score: '      f'{valScr             :6.3f}', end = '')
            print(' | Epoch Time: '     f'{epochTime          :5.2f}', end = '')

            # Save best model ("Early Stopping")
            if valScr > bestScore:
                bestScore = valScr
                print(' | <-- Checkpoint!', end = '')
                try:
                    dCheckpoint = {'Model' : oModel.state_dict(), 'Optimizer' : oOpt.state_dict(), 'Scheduler': oSch.state_dict()}
                    torch.save(dCheckpoint, 'BestModel.pt')
                except:
                    print(' | <-- Failed!', end = '')
            print(' |')
        
        # Load best model ("Early Stopping")
        dCheckpoint = torch.load('BestModel.pt')
        oModel.load_state_dict(dCheckpoint['Model'])

        return oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate



        