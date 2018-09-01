import pandas as pd
import sys, os
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import scipy.io as sio
import threading, subprocess
from random import randint
import sched, time as Time
import pickle
from subprocess import Popen, PIPE
#import matlab.engine

# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

count1 = 0

FILEPATH = os.path.abspath(__file__)

ECGdata = []
GSRdata = []

data = [GSRdata, ECGdata]

ECGtime = []
GSRtime = []

#(RTStream)
#For simulting Real Time data
RTStreamTIME = np.array([])
RTStreamECG = np.array([])
RTStreamGSR = np.array([])
RTStreamEVENTS = np.array([])

#(RTStream)
#Pointer to the current data streamed
pointer = 0

#Stream
DataStream = np.zeros(3) #The data stream comes here: Data[0] = ecg val, Data[1] = gsr val, Data[2] = event/notEvent
TimeStream = np.array([0])

#(RTUpdateSignals)
#Pointer to the current data added from stream
index = 0
#The sequence data and time
RTtime = []
RTdata = [[], []]
#predictions to current block
blockPredictions = []
block_num = 0
# Counting any seg_len reading data in the same block
counter_seg = 0
#flag if we reading data from block
reading_block = 0

time = [GSRtime, ECGtime]



#Events
start_block = 1 #'run_start'
end_block = 2 #'run_end'
end_data = 3 #'finish'
events = [start_block, end_block, end_data]

#ML
NETWORK = 0 #Use cllasifier or neural network
filename = 'finalized Gradient Boosting Classifier.sav'

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

seg_len = 2200

ecg_features = []
gsr_features = []

tags_times = []

##############################

# Run .mat scripts
#eng = matlab.engine.start_matlab()

def makeBlocksList(tags, tags_time):
    lst = []
    for i in range(len(tags)):
        if (tags[i] == start_block): lst.append([tags_times[i], ])
        if (tags[i] == end_block): lst[-1].append(tags_times[i])
    return lst

def clearSignal(signal):
    clearedSignal = []
    return clearedSignal


#################################
def RTupdate():
    #print('RTupdate')
    RTStream() #get current data: ecg, gsr and event and updates in DataStream
    RTupdateSignals()

########## methods for real time grahps plotting and predictions ############

#Simulates Real Time streaming data
#Takes from global variables: RTStreamECG, RTStreamGSR, RTStreamEVENTS, RTStreamTIME, DataStream
#Put in DataStream the data
def RTStream():
    global RTStreamECG, RTStreamGSR, RTStreamEVENTS, RTStreamTIME, DataStream, TimeStream, end_data, pointer

    #Streaming new data
    np.put(DataStream, [0,1,2], [RTStreamECG[pointer], RTStreamGSR[pointer], RTStreamEVENTS[pointer]])
    np.put(TimeStream, 0, RTStreamTIME[pointer])

    #Finish reading all data
    if RTStreamEVENTS[pointer] == end_data:
        #finish = True
        print('DataStream: exit')

    #Update the pointer
    pointer += 1
    #check
    #print('DataStream: done read')


#input: signal and flag specify the data type.
#flag = 0 <=> is ecg, flag = 1 <=> is gsr
#Excecute .mat script to extract features from signal flag: 0 if ecg. 1 if gsr
def extractSeg(signal, flag):
    print('extractSeg')

    # if NETWORK == 1:
    #     input_seg = eng.mat_signals(signal, flag)  # Run .mat script to filter signals
    #     X = input_seg
    #     return input_seg
    # else:
    #     #Extract features from signal: ECG or GSR
    #     input_features = eng.mat_features(signal, flag)  # Run .mat script to extract features from signals
    #     if flag == 0: #ecg
    #         ecg_features = input_features
    #
    #     else: #flag == 1: gsr
    #         gsr_features = input_features

#Gets inputs and loaded model
#inputes : array's length = 3. inputes[0] = ecg, inputes[1] = gsr, inputes[2] = events
#loaded_model for prediction
#Predict Cl of segment and return result.
#The function excecute two threads for calculate features of gsr and ecg parallel
def predictSeg(inputs):
    #Check:'''''''''''''
    filename_glass = r'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\data.csv'
    df_data = pd.read_csv(filename_glass)

    # Take only levels in: levels
    levels = [1, 2, 3, 4, 5]
    df_data = df_data.loc[df_data['level'].isin([1,2,3,4,5])]
    y_cols = 'level'
    x_cols = list(df_data.columns.values)
    x_cols.remove(y_cols)
    features = df_data.iloc[0]
    #"""""""""""""""""""""""""""""""""""""""""""""""


    # # start extracting features from ECG on new thread
    # t0 = threading.Timer(0, extractSeg, (inputs[0], 0), )
    # t0.start()
    #
    # # start extracting features from GSR on new thread
    # t1 = threading.Timer(0, extractSeg, (inputs[1], 1), )
    # t1.start()
    #
    # # Wait for extracting all features: ECG and GSR
    # t0.join()
    # t1.join()
    #
    # #Concat the features
    # features = [ecg_features + gsr_features]
    #
    # #Clear the features
    # ecg_features.clear()
    # gsr_features.clear()

    # predict the result of the segment
    features = np.asarray(features)
    features = features.reshape(1, -1)  #Maybe need
    result = loaded_model.predict(features)
    print('predictSeg : result is:',result) #For check

    return result


#Update in RT the signals plots and calling the predictSeg when needed
#Calculate the final predict of the block
def RTupdateSignals():

    global DataStream, TimeStream, index, counter_seg, reading_block, RTtime, RTdata, blockPredictions, block_num

    #When we update the plots of the data streaming
    display_jump = 10

    #flag end of streaming data
    # finish = False

    # Insert to RTData array ecg and gsr values from the data stream
    RTdata[0].append(DataStream[0])
    RTdata[1].append(DataStream[1])
    RTtime.append(TimeStream[0])

    if counter_seg == seg_len: #Finish reading segment
        print('Finish reading segment block number:', block_num)

        segment = [RTdata[0][index-counter_seg+1:index+1],RTdata[1][index-counter_seg+1:index+1]]
        print('segment', type(segment))

        #@Real code
        result = predictSeg(segment)
        blockPredictions.extend(result) #Add predict to list of segmentations' predicts in the block
        print('result for seg:',result)

        counter_seg = 0 #Reset counter
        block_num += 1 #count another block

        #@OR: need to display what is the prediction for this current segment

    # Finish reading block data.
    # Predict the block:
    if DataStream[2] == end_block:
        # OR: add label specify it's end of the block
        print('Finish reading block data. Need to predict the block')
        block_predict = int(np.median(blockPredictions))
        print('Predict of the block',block_num,'is:',block_predict)


        #@Real code
        # block_predict = int(np.median(blockPredictions))
        # print('from RTupdateSignals: Predict is: ', block_predict) #For check
        # #Clear predictions
        # blockPredictions.clear()
        # #Reset counter
        # counter_seg = 0

        #Or:
        #Do something with predict: on the graph show
        #Display what we predict for the block

    elif DataStream[2] == start_block: #Starting reading new block's data
        print('Starting reading new blocks data')
        #OR: add label specify it's start of block
        reading_block = 1 #Change flag

    if  reading_block == 1:
        #print('Update the counter_seg')
        #Update the counter_seg
        counter_seg += 1

    #Display plot
    if index % display_jump == 0 and index > 0:
        ECGcurve.setData(np.asarray(RTtime), np.asarray(RTdata[0]))
        GSRcurve.setData(np.asarray(RTtime), np.asarray(RTdata[1]))
        QtGui.QApplication.processEvents()

    if DataStream[2] == end_data:
        print('finish') #check
        #finish = True

    # Update index
    index += 1
#''Need to add return?

############## methods for GUI interfence ###################


class runBlack(QDialog):
    def __init__(self):
        super(runBlack, self).__init__()
        loadUi('runBlack.ui', self)
        self.setWindowTitle('hello world')
        self.loadGSR.clicked.connect(self.loadGSR_clicked)
        self.loadECG.clicked.connect(self.loadECG_clicked)
        self.RTinterface.clicked.connect(self.RTinterface_clicked)
        self.start.clicked.connect(self.start_clicked)
        self.reset.clicked.connect(self.reset_clicked)

    @pyqtSlot()


#Real Time mode: load data
    def RTinterface_clicked(self):
        if self.rtMode.isChecked() == True:

            #global ECGdata, data, GSRdata, ECGtime, GSRtime, time, GSRcurve, ECGcurve, p
            global ECGcurve, GSRcurve
            global RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS

            p = self.rawDataView.addPlot()
            #p = self.rawDataView.addPlot(downsample=4)

            # p.setXRange(-1.06, 878.15)
            # p.setYRange(4.82, 9.3)

            # set properties
            #Need to change a bit the units scale
            #Need to make to windows for gsr and ecg seperate
            p.setLabel('left', 'Value', units='V')
            p.setLabel('bottom', 'Time', units='s')
            p.setXRange(0, 5000)
            p.setYRange(-600, 600)
            p.setWindowTitle('pyqtgraph plot')
            p.enableAutoScale()

            # plot
            ECGcurve = p.plot(pen='r')
            GSRcurve = p.plot(pen='b')

            # c1 = plt.plot(x, y, pen='b', symbol='x', symbolPen='b', symbolBrush=0.2, name='red')
            # c2 = plt.plot(x, y2, pen='r', symbol='o', symbolPen='r', symbolBrush=0.2, name='blue')



            print('RTinterface_clicked: ECGcurve',ECGcurve )
            print('RTinterface_clicked: ECGcurve.shape', ECGcurve.shape)
            print('RTinterface_clicked: GSRcurve',GSRcurve )
            print('RTinterface_clicked: GSRcurve.shape', GSRcurve.shape)

            fgsr = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
            m = sio.loadmat(fgsr)
            # RTStreamTIME = m['time_index'][14000:]
            # RTStreamECG = m['ecg'][14000:]
            # RTStreamGSR = m['gsr'][14000:]
            # RTStreamEVENTS =  m['events'][14000:]
            RTStreamTIME = m['time_index'][13240:]
            RTStreamECG = m['ecg'][13240:]
            RTStreamGSR = m['gsr'][13240:]
            RTStreamEVENTS =  m['events'][13240:]
            #check
            print('RTinterface_clicked: added all data for streaming')
            print('RTinterface_clicked: RTStreamTIME.shape',RTStreamTIME.shape)
            print('RTinterface_clicked: RTStreamTIME', RTStreamTIME)
            print('RTinterface_clicked: RTStreamECG.shape',RTStreamECG.shape)
            print('RTinterface_clicked: RTStreamECG', RTStreamECG)
            print('RTinterface_clicked: RTStreamGSR.shape',RTStreamGSR.shape)
            print('RTinterface_clicked: RTStreamGSR', RTStreamGSR)
            print('RTinterface_clicked: RTStreamEVENTS.shape',RTStreamEVENTS.shape)
            print('RTinterface_clicked: RTStreamEVENTS', RTStreamEVENTS)
            print('RTinterface_clicked: RTStreamEVENTS type', type(RTStreamEVENTS))
            #print('RTStreamTIME', RTStreamTIME)
            # del m

#######################################################

#Or:
#Static mode: load data
    def loadECG_clicked(self):

        # self.rawDataView.clear()
        global ECGdata, data, GSRdata, ECGtime, GSRtime, time, ECGcurve, start_time
        fecg = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        '''m= sio.loadmat(fecg)
        tmp = m['time_index'][0]
        ECGtime = []
        for t in tmp: ECGtime += [t/1000]
        ECGdata = m['data'][0]
        del m'''
        ECGtime = GSRtime
        ECGdata = np.random.normal(loc=6.5, scale=0.3, size=len(GSRtime))
        data = [GSRdata, ECGdata]
        time = [GSRtime, ECGtime]

        if self.rtMode.isChecked() == True:
            # load ECG clicked when the system is on real time mode #
            '''sync_time = count1 # = GSR current position
            start_time=Time.time()
            threading.Timer(0, updateECGplot, (start_time,sync_time,)).start() #start ploting ECG on new thread'''

        else:
            # load ECG clicked when the system is on static test mode #
            ECGcurve.setData(ECGtime, ECGdata)


    def loadGSR_clicked(self):

        global ECGdata, data, GSRdata, ECGtime, GSRtime, time, GSRcurve, ECGcurve, p

        if GSRdata == []:
            p = self.rawDataView.addPlot()
            ECGcurve = p.plot(pen=(0, 2))
            GSRcurve = p.plot(pen=(1, 2))

        fgsr = QFileDialog.getOpenFileName(self, 'Open file', "mat files (.mat)")[0]
        m = sio.loadmat(fgsr)
        GSRtime = m['time_index'][0]
        GSRdata = m['data'][0]
        del m
        data = [GSRdata, ECGdata]
        time = [GSRtime, ECGtime]
        p.setXRange(-1.06, 878.15)
        p.setYRange(4.82, 9.3)

        if self.rtMode.isChecked() == True:
            # load ECG clicked when the system is on real time mode #
            '''starttime=Time.time()
            threading.Timer(0, updateGSRplot, (starttime,)).start() #start ploting GSR on new thread'''

        else:
            # load GSR clicked when the system is on static test mode #
            GSRcurve.setData(GSRtime, GSRdata)
#######################################################


    #START
    def start_clicked(self):

        global data, ECGdata, GSRdata

        #RT variables for simulating streaming
        global RTStreamTIME, RTStreamECG, RTStreamGSR, RTStreamEVENTS

        # Real Time mode
        if self.rtMode.isChecked() == True:
            # The data that will be used to simulate streaming is loaded
            #if ((RTStreamTIME != []) and (RTStreamECG != []) and (RTStreamGSR != []) and (RTStreamEVENTS != [])):
                print('started')
                self.timer = pg.QtCore.QTimer()
                self.timer.timeout.connect(RTupdate)
                self.timer.start(10)

        #Or:
        else:
            # system is on static test mode #
            data = [GSRdata, ECGdata]
            p1 = self.modDataView.addPlot(x=time[1], y=data[1], pen=(1, 2))
            p2 = pg.PlotCurveItem(x=time[0], y=data[0], pen=(0, 2))
            p1.addItem(p2)

            # for i in range(len(blocks)):
            #     # add the network workeloads predictons of each data block to the graphic view
            #     level = predictBlock(GSRdata[blocks[i][0]:blocks[i][1]], ECGdata[blocks[i][0]:blocks[i][1]])
            #     lr = pg.LinearRegionItem(values=[150 * (i), 150 * (i + 1)], brush=pg.intColor(index=level, alpha=50),
            #                              movable=False)
            #     p1.addItem(lr)
            #     label = pg.InfLineLabel(lr.lines[1], "oveload " + str(level), position=0.85, rotateAxis=(1, 0),
            #                             anchor=(1, 1))
            #
            # else:
            #     # either GSR or ECG files has not been loaded
            #     print('error')


    def reset_clicked(self):
        try:
            subprocess.Popen([sys.executable, FILEPATH])
        except OSError as exception:
            print('ERROR: could not restart aplication:')
            print('  %s' % str(exception))
        else:
            QApplication.quit()

def run():
    #Start App
    app = QApplication(sys.argv)
    widget = runBlack()
    widget.show()
    sys.exit(app.exec_())

run()

#
# #-*- coding: utf-8 -*-
# import random
# import time
# from collections import deque
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# import numpy as np
# import os
# import spidev
#
# win = pg.GraphicsWindow()
# win.setWindowTitle('DOTS')
#
#
# p1 = win.addPlot()
# p1.setRange(yRange=[0,25])
# p1.setRange(xRange=[0,25])
# curve1 = p1.plot()
#
#
# nsamples=300 #Number of lines for the data
#
# dataRed= np.zeros((nsamples,2),float) #Matrix for the Red dots
# dataBlue=np.zeros((nsamples,2),float) #Matrix for the Blue dots
#
# def getData():
#     global dataRed, dataBlue
#
#     t0= random.uniform(1.6,20.5) #Acquiring Data
#     d0= random.uniform(1.6,20.5) #Acquiring Data
#     vec=(t0, d0)
#
#     dataRed[:-1] = dataRed[1:]
#     dataRed[-1]=np.array(vec)
#
#     t0= random.uniform(1.6,20.5) #Acquiring Data
#     d0= random.uniform(1.6,20.5) #Acquiring Data
#     vec=(t0, d0)
#
#     dataBlue[:-1] = dataBlue[1:]
#     dataBlue[-1]=np.array(vec)
#
#
# def plot():
#
#     #Blue Dots
#     curve1.setData(dataBlue, pen=None, symbol='o', symbolPen=None, symbolSize=4, symbolBrush=('b'))
#     #Red Dots
#     curve1.setData(dataRed, pen=None, symbol='o', symbolPen=None, symbolSize=4, symbolBrush=('r'))
#
#
# def update():
#
#     getData()
#     plot()
#
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)
#
# ## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()# -*- coding: utf-8 -*-
