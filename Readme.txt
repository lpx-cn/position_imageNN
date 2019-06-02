Using CNN and ResNet to locate the camera.

1. photos_generate.py

	Can obtain the trainning data (uniform or random) with any camera.

		Parameters: room()--room size, LED's position, LED's power (the radius of the sphere) 
					camera()--cameras'angle, focal, sensor size, resolution
		Functions:	camera()--move, rotate, take_photo, __euler2direction() for obtain the camera's angle
	   						__pixelvalue() for obatain the coordinate of the PD's in WCS.
							isInside() for judge if the PD can receive the signal.
							data_generate_uniformity()
							data_generate_random()
2. cnn_position.py
	
	Can obtain the CNN model for camera positioning (with low resolution).

		Parameters: path of dataset and model saving, some neural network parameters(batch size, optimizer, epoch, learning rate reducer and so on)  
		Functions:	mkdir() for generate the path
					MPE() define the mean positioning error function for selecting the model
					_bn_relu() and _conv_bn_relu() build some custom layers
					keras_debug() obtain the training data and validation data, build the network, then train and save model 

3. cnn_random_position.py

	Can obtain the CNN model which can predict camera's position and angle (with low resolution).

		Parameters: same as cnn_position.py
		Functions: keras_debug() normalize the output.
					others are same as cnn_position.

4. resnet.py

	Build the ResNet network. The output is the camera's position.

5. resnet_AP.py

	Build the ResNet network, which is used for predicting the camera's position and angle. Thus, there are two outputs.

6. resnet_position.py
	
	Can obtain the ResNet model for camera positioning (with high resolution).

	Except using resnet.py to build the network, it's samiliar as cnn_position.py.	

7. resnet_random_position.py

	Can obtain the ResNet model which can predict camera's position and angle (with high resolution).

	Except using resnet_AP.py to build the network, it's samiliar as cnn_random_position.py

8. model_evaluate.py

	Can evaluate the all model in any path.

9. height_MPE.py
	
	 Can obtain the photo of "camera's height--MPE" from a model.
	
Folder:	dataset--all training and test data. (28/224, uniform/random, with angle/without angle)
		debug--all results from different models. (model struction, model weights, MPE, training log)
		height-MPE--the relationship between cameras's height and mean positioning error.
	   	testmodel--many models for evaluation
		log-- python -u *.py | tee ./log/*.log
		photo_result--the heigth-MPE photos of all models.
