# LANE DETECTION ALGORITHM

Aim: To build an algorithm which can detect lanes in a road and calculate and display features like offset from the center of the lane, radius of curvature of the road etc.

Methodology:

Part 1: Identification of the Lanes

CNN MODEL: SegNet architecture is used with optimized modifications to it. Encoder-Decoder pairs are used to create feature maps for classifications of different resolutions. The architecture is designed such that- the input is an image of road, the labeled output is an image with the lane marked with a single RGB channel, hence the output of the CNN model is an image with lane marked in a single RGB channel.

DATA SET: 13,000 inputs and labels (80 x 160 dimensions)

TOOLS/PACKAGES:  Keras, Scikit-learn;  LANGUAGE: Python

ARCHITECTURE:
•	Encoding layer:  7 layers of Convolution operation along with intermittent Maxpooling and Dropouts
•	LSTM layer: 2 LSTM layers, resizing performed after each layer to maintain matrix dimensions
•	Decoding layer: 7 layers of transpose of Convolution operation or deconvolution performed with intermittent Upsampling and Dropouts
•	Adam Optimizer and Mean squared loss used 

RESULTS

Batch Size: 128, Epochs: 10
Accuracy without LSTM layers=92% 
Accuracy with LSTM layers=93.5% 

Part 2: Mathematical Calculation and Extraction of Offset and Radius of Curvature of road 

MODEL: Use the live camera/video feed obtained after running the trained parameters of the CNN model which will give us the identified lane.  Extract the required features using OpenCV functions and mathematical formulas.

TOOLS/PACKAGES:  OpenCV,   LANGUAGE: Python

ARCHITECTURE:
•	Perform edge detection to get the edges of the lane and perspective transform to obtain perpendicular view of the lane
•	Identify the two separate lanes using sliding window technique
•	Use polynomial fit curve function to find out the quadratic function of curve of the lane
•	Once we have the quadratic function for the curvature of the lane, the radius of curvature of the road and offset from center of the road is calculated by mathematical formulas.

RESULTS
Offset calculation: giving result within 1-2m range which is ideal
Radius of curvature: two radius of curvatures obtained for left and right boundary of lane each. Values correspond with the visual curves of the road.

CONCLUSION
This is a working lane detection algorithm model which seems to give a fair output on the recorded videos of the road. Real time deployment of this model is needed to verify and fine tune the mathematical calculations done in feature extraction.

