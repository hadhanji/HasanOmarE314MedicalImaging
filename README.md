This project uses Convolution Neural Network to train a model which classifies chest scan images of Covid, Viral Pneumonia, and Normal people. The original images are run through the code and the result is recorded. Then the images are resized so that a discrete Fourier Transform can be completed on them. Once the Transform is done, the transformed images are run through the original image code.

# HasanOmarE314MedicalImaging
The demonstration video is shown in the Google Drive link below:
https://drive.google.com/file/d/1cv-Qx14Ul3evHhwIjiPPO5_xEsx8AQ6O/view?usp=sharing

The following resources were used as references for this project:

CNN Model With PyTorch For Image Classification by Pranjal Soni: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
Kaggle Notebook used as reference from the article above:
https://www.kaggle.com/pranjalsoni17/natural-scene-classification/notebook?scriptVersionId=51522484&cellId=33

Resizing multiple images and saving them using OpenCV by Basit Javed:
https://medium.com/@basit.javed.awan/resizing-multiple-images-and-saving-them-using-opencv-518f385c28d3

OpenCV Python Tutorials: Fourier Transform:
https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

The “untitled0.py” file is the code run on the original images with no Fourier Transform. 

“Resize.py” is the code used to resize the images in each directory. The code was run multiple times with different input and output filenames specified for each class in the test and training sets. 

“DTFT.py” performs the Fourier transform on the images and outputs the transform to the specified folder. This code was also run multiple times for each class in the test and training sets with different input and output locations specified.

“FourierUntitled0” is the same as untitled0 but changed so that it would work for our purposes.

Moving Forward:

The Fourier Results are not what we expected them to be and that is likely due to the fact that the transform ignored the phase and is only concerned with the magnitude. A next step would likely be to somehow include the phase of the DTFT in the output images instead of just the magnitude. This would likely get better results.

Another step would be to somehow combine both the DTFT and the original images as this could yield a higher accuracy than both of them separately. 

Contributions:

Omar: Completed the resizing and Fourier transform of the images, and adapted the code for the original images so that it would work for the Fourier images.

Hasan: Completed the code for the original images and put everything together. Helped with the Fourier Transform code.
