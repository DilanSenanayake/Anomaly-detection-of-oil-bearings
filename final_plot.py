import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFrame, QMessageBox
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import TestPlot
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

dataset = pd.read_csv("C:\\Users\\Dilan\\Desktop\\GUI\\dataset.csv")    
      
label_set = np.array(dataset.loc[:, 'label'])

class CNN(nn.Module):
      def __init__(self):
          super(CNN, self).__init__()
          self.conv1 = nn.Conv2d(4, 32, kernel_size=4,stride=1,padding = 1)
          self.mp1 = nn.MaxPool2d(kernel_size=2,stride=1)
          self.conv2 = nn.Conv2d(32,64, kernel_size=4,stride =1,padding = 1)
          self.mp2 = nn.MaxPool2d(kernel_size=2,stride=1)
          self.fc1= nn.Linear(2304,256)
          self.dp1 = nn.Dropout(p=0.4)
          self.fc2 = nn.Linear(256,4)

      def forward(self, x):
          in_size = x.size(0)
          x = F.relu(self.mp1(self.conv1(x)))    
          x = F.relu(self.mp2(self.conv2(x)))
          x = x.view(in_size,-1)
          x = F.relu(self.fc1(x))
          x = self.dp1(x)
          x = self.fc2(x)
          
          return F.log_softmax(x, dim=1)

cnn = CNN().double()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)

# Load the saved model
cnn = torch.load('G:/My Drive/KCCP/Model_03.pth')

def result(k):
  datasetA = dataset.loc[:, 'bb10']
  datasetB = dataset.loc[:, 'TNH']
  datasetC = dataset.loc[:, 'DWATT']
  datasetD = dataset.loc[:, 'btgj1']
  label = np.array([label_set[k]])

  # A - bb10
  a = 0.001576136914
  l = 0.1803473532 - 0.001576136914

  r = k
  p = []
  for j in range(100):
    p.append((datasetA[r]-a)/l)
    r = r + 1
  dataA = np.array([p])


  # B - TNH
  a = 33.79629898
  l = 102.0188599 - 33.79629898

  r = k
  p = []
  for j in range(100):
    p.append((datasetB[r]-a)/l)
    r = r + 1
  dataB = np.array([p])


  # C - DWATT
  a = 0.1725509763
  l = 109.9144211 - 0.1725509763

  r = k
  p = []
  for j in range(100):
    p.append((datasetC[r]-a)/l)
    r = r + 1
  dataC = np.array([p])
  

  # D - btgj1
  a = 166.0964661
  l = 196.0968933 - 166.0964661

  r = k

  p = []
  for j in range(100):
    p.append((datasetD[r]-a)/l)
    r = r + 1
  dataD = np.array([p])

  def sig_image(data,size):
      X=np.zeros((data.shape[0],size,size))
      for i in range(data.shape[0]):
          X[i]=(data[i,:].reshape(size,size))
      return X.astype(np.float16)

  # bb10
  x_A = sig_image(dataA,10)

  # TNH
  x_B = sig_image(dataB,10)

  # DWATT
  x_C = sig_image(dataC,10)

  # btgj1
  x_D = sig_image(dataD,10)

  
  X = np.stack((x_A,x_B,x_C,x_D),axis=1).astype(np.float16)
#   print(X.shape)

  sig_test = torch.from_numpy(X)
  lab_test = torch.from_numpy(label)

  import torch.utils.data as data_utils
  batch_size = 512
  test_tensor = data_utils.TensorDataset(sig_test, lab_test) 
  test_loader = data_utils.DataLoader(dataset = test_tensor, batch_size = batch_size, shuffle = False)

  

  total_step = len(test_loader)
#   print(total_step)
  loss_list_test = []
  acc_list_test = []
  with torch.no_grad():
      for i, (signals, labels) in enumerate(test_loader):
          # Run the forward pass
          #print(labels)
          signals=signals
          labels=labels
          outputs = cnn(signals.double())
          loss = criterion(outputs, labels.long())
          loss_list_test.append(loss.item())
          # if epoch%10 ==0:
          #     print(loss)
          total = labels.size(0)
          _, predicted = torch.max(outputs.data, 1)
          correct = (predicted == labels.long()).sum().item()
          acc_list_test.append(correct / total)
          #print(outputs.shape)
          #print(outputs)
          print(predicted.item())
          #print(labels.shape)
          # if (epoch) % 1 == 0:
          #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
          #                   (correct / total) * 100))
  return(predicted.item())



# print(result(k))








class PlotFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a figure and axes for the plot
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)

        # Create the canvas to display the plot
        self.canvas = FigureCanvas(self.figure)

        # Create the main layout and add the canvas to it
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        # Set the layout for the frame
        self.setLayout(layout)

        # Generate initial plot
        # self.update_plot(0)

    
        # Vibration plot
    def update_plot1(self,counter):
        # Clear the previous plot
        self.axes.clear()

        

        # Generate new data for the plot
        arr1 = TestPlot.plot1(counter)
        x = arr1[0]
        y = arr1[1]
        
        # Plot the data
        self.axes.plot(x, y)

        # Add axis labels and title
        self.axes.set_xlabel('Time (s) [span: 60s]')
        self.axes.set_ylabel('Vibration (mm/s^2)')
        self.axes.set_title('Vibrations of the bearing')

        # Hide x-axis tick values
        self.axes.set_xticks([])
        
        # Add grid
        self.axes.grid(True)

        # Refresh the canvas
        self.canvas.draw()

    # rpm plot
    def update_plot2(self,counter):
        # Clear the previous plot
        self.axes.clear()

        

        # Generate new data for the plot
        arr1 = TestPlot.plot2(counter)
        x = arr1[0]
        y = arr1[1]
        
        # Plot the data
        self.axes.plot(x, y)

        # Add axis labels and title
        self.axes.set_xlabel('Time (s) [span: 60s]')
        self.axes.set_ylabel('RPM %')
        self.axes.set_title('RPM of the generator')

        # Hide x-axis tick values
        self.axes.set_xticks([])
        
        # Add grid
        self.axes.grid(True)

        # Refresh the canvas
        self.canvas.draw()

    # temp plot
    def update_plot3(self,counter):
        # Clear the previous plot
        self.axes.clear()

        

        # Generate new data for the plot
        arr1 = TestPlot.plot3(counter)
        x = arr1[0]
        y = arr1[1]
        
        # Plot the data
        self.axes.plot(x, y)

        # Add axis labels and title
        self.axes.set_xlabel('Time (s) [span: 60s]')
        self.axes.set_ylabel('Temperature (`C)')
        self.axes.set_title('Temperature of oil')

        # Hide x-axis tick values
        self.axes.set_xticks([])
        
        # Add grid
        self.axes.grid(True)

        # Refresh the canvas
        self.canvas.draw()
    

    # active power plot
    def update_plot4(self,counter):
        # Clear the previous plot
        self.axes.clear()

        

        # Generate new data for the plot
        arr1 = TestPlot.plot4(counter)
        x = arr1[0]
        y = arr1[1]
        
        # Plot the data
        self.axes.plot(x, y)

        # Add axis labels and title
        self.axes.set_xlabel('Time (s) [span: 60s]')
        self.axes.set_ylabel('Active power (MW)')
        self.axes.set_title('Active power of the generator')

        # Hide x-axis tick values
        self.axes.set_xticks([])
        
        # Add grid
        self.axes.grid(True)

        # Refresh the canvas
        self.canvas.draw()

A_1 = []
A_2 = []
A_3 = []

from PyQt5.QtCore import QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        loadUi("Test1.ui", self)

        # Find the frame_3, frame_4, frame_5, and frame_8 widgets from the loaded UI
        frame_3 = self.findChild(QFrame, "frame_3")
        frame_4 = self.findChild(QFrame, "frame_4")
        frame_5 = self.findChild(QFrame, "frame_5")
        frame_8 = self.findChild(QFrame, "frame_8")

        # Create the plot frames
        self.plot_frame_3 = PlotFrame(frame_3)
        self.plot_frame_4 = PlotFrame(frame_4)
        self.plot_frame_5 = PlotFrame(frame_5)
        self.plot_frame_8 = PlotFrame(frame_8)

        # Set the plot frames as the central widgets for the respective frames
        frame_3.setLayout(QVBoxLayout())
        frame_3.layout().addWidget(self.plot_frame_3)
        frame_4.setLayout(QVBoxLayout())
        frame_4.layout().addWidget(self.plot_frame_4)
        frame_5.setLayout(QVBoxLayout())
        frame_5.layout().addWidget(self.plot_frame_5)
        frame_8.setLayout(QVBoxLayout())
        frame_8.layout().addWidget(self.plot_frame_8)

        # Create a QTimer for updating the plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1000)  # Update every 1000 milliseconds (1 second)

        # Initialize the variable to be incremented
        self.counter = 0


    def update_plots(self):
        # Increment the variable
        self.counter += 50

        # Update the plots in each plot frame
        self.plot_frame_3.update_plot1(self.counter)
        self.plot_frame_4.update_plot2(self.counter)
        self.plot_frame_5.update_plot3(self.counter)
        self.plot_frame_8.update_plot4(self.counter)

        # Update the color of QLabel_1 based on the condition
        # print(TestPlot.label())
        # if TestPlot.label(self.counter) == 0:  # Replace with your own condition
        if result(self.counter) == 0:
            self.label_8.setStyleSheet("background-color: green;")
            self.label_6.setStyleSheet("")
            self.label_4.setStyleSheet("")
            self.label_3.setStyleSheet("")
        
        if result(self.counter) == 3:
            self.label_6.setStyleSheet("background-color: yellow;")
            self.label_8.setStyleSheet("")
            self.label_4.setStyleSheet("")
            self.label_3.setStyleSheet("")
            if len(A_1) == 0:
                QMessageBox.information(self, "Warning!", "Anomaly level 1 detected")
                A_1.append(1)
        if result(self.counter) == 2:
            self.label_4.setStyleSheet("background-color: orange;")
            self.label_6.setStyleSheet("")
            self.label_3.setStyleSheet("")
            self.label_8.setStyleSheet("")
            if len(A_2) == 0:
                QMessageBox.information(self, "Warning!", "Anomaly level 2 detected")
                A_2.append(2)
        if result(self.counter) == 1:
            self.label_3.setStyleSheet("background-color: red;")
            self.label_4.setStyleSheet("")
            self.label_6.setStyleSheet("")
            self.label_8.setStyleSheet("")
            if len(A_3) == 0:
                QMessageBox.information(self, "Warning!", "Anomaly level 3 detected")
                A_3.append(3)
        




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
