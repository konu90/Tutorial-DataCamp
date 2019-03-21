# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
hello = tf.constant('Hello, World')
sess = tf.Session()
print(sess.run(hello))

print("This is a basic tensor tutorial of TensorFlow from DataCamp: https://www.datacamp.com/community/tutorials/tensorflow-tutorial")

# +
# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])
#Existen dos tipos mas de valores aparte de las constantes, que son los placeholders y las variables.
#Los Placeholders son valores que no estan asignados y que seran inicializados cuando se ejecute la sesion.
#el tipo placeholder viene a ser como un marcador de posicion para un tensor que es inicializado por la sesion cuando
#esta se ejecuta
#Por otro lado, esta el tipo "variable", que son valores que se pueden modificar,y las constantes son constantes -_-!
# Multiply
result = tf.multiply(x1, x2)

pantalla1 = tf.constant(["Opcion 1:"], tf.string)
pantalla2 = tf.Variable(["Opcion 2:"], tf.string)
# Print the result
print(result)
print(pantalla1)
print(pantalla2)
#Por pantalla se muestra un tensor abstracto en el "computational graph", pero la operación no se ha ejecutado, 
#se ha definido el modelo. TensorFlow sigue el modelo "lazy evaluation" o evaluacion perezosa

# +
# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

pantalla1 = tf.constant(["Opcion 1:"], tf.string)
pantalla2 = tf.constant(["Opcion 2:"], tf.string)
pantalla1 = tf.constant(["Opcion buba:"], tf.string)

#Para mostrar el resultado del tensor, debemos de inicializar o ejecutar la sesion. Existen dos maneras. Una debemos de 
#cerrarla nosotros, y otra se cierra automaticamente al ejecutarse la ultima linea que contiene. Aunque en el tutorial 
#de datacamp son dos ejecuciones diferentes, y la sesion es cerrada, ejecutare aqui ambas opciones.

############Opcion 1########
#Debemos de cerrar nosotros la sesion
# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(pantalla1))
print(sess.run(result))
# Close the session
sess.close()

###########Opcion 2##########
#La sesion se cierra automaticamente

# Initialize Session and run `result`
with tf.Session() as sess:
    #Nota aqui no podemos hacer lo siguiente:
#     pantalla = sess.run(pantalla2)
#     print(pantalla)
    #Esto da un fallo, porque la variable no ha sido inicializada. Las variables en tensorflow deben de ser inicialiadas
    #antes de poder usarlas, para ello, tenemos dos opciones, o inicializar una variable solo(o conjunto de variables)
    #o inicializar directamente todas las variables, aunque he probado no me ha salido, asi que se vera mas adelante
    pantalla = sess.run(pantalla2)
    print(pantalla)
    print(sess.run(pantalla2))
    output = sess.run(result)
    print(output)

# +
#En los ejemplos anteriores, hemos definido una sesion predeterminada, pero a una sesion tambien se le pueden pasar
#opciones de configuracion
# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(result))
# sess.close()
# # Initialize Session and run `result`
# with tf.Session() as sess:
#     config=tf.ConfigProto(log_device_placement=True)
#     output = sess.run(result)
#     print(output)
# #     print(config)


# +
#This part is from official tensorflow page: 
# https://www.tensorflow.org/guide/using_gpu

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
#We should be something similar to:
##### Device mapping:
##### /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
##### id: 0000:05:00.0
##### b: /job:localhost/replica:0/task:0/device:GPU:0
##### a: /job:localhost/replica:0/task:0/device:GPU:0
##### MatMul: /job:localhost/replica:0/task:0/device:GPU:0
##### [[ 22.  28.]
#####  [ 49.  64.]]
# But this not works for me.


# +
#Other way to see info from run and CPU device executer
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session()

# Runs the op.
options = tf.RunOptions(output_partition_graphs=True)
metadata = tf.RunMetadata()
c_val = sess.run(c, options=options, run_metadata=metadata)

print(metadata.partition_graphs)
# -

############################ BELGIAN TRAFFIC SIGNS ##########################################
print("This is a tutorial of neural network with TrafficSigns from DataCamp in tensorflow:")
print("https://www.datacamp.com/community/tutorials/tensorflow-tutorial")

# Download files to use in this tutorial
# Training and testing datasets
# https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip
# https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip
########## IMPORTS TO THIS BELGIAN TRAFFIC SIGNS TUTORIAL ######
#Import tensorflow
# import tensorflow as tf
import skimage
#Import the `transform` module from `skimage`
# Problems with version, we should do "pip install --upgrade scikit-image" in anaconda promt in order to upgrade packages
from skimage import transform
#scikit-image is an image processing Python package that works with numpy arrays. 
#The package is imported as skimage:
#import skimage
# Most functions of skimage are found within submodules:
from skimage import data
# Import the `pyplot` module to paint graphs
import matplotlib.pyplot as plt
#El módulo os nos permite acceder a funcionalidades dependientes del Sistema Operativo. 
#Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos permiten manipular 
#la estructura de directorios (para leer y escribir archivos)
import os
#Numpy
import numpy as np
# Convert to grayscale
from skimage.color import rgb2gray
import random
# import sys
# print(sys.path)

# +
#Function to load_data
#Note: You’ll see that the testing, as well as the training data folders, contain 61 subfolders,
#which are the 62 types of traffic signs that you’ll use for classification in this tutorial. 
#Additionally, you’ll find that the files have the file extension .ppm or Portable Pixmap Format.
#We pass the train_data_directory to the function.
def load_data(data_directory):
    
    #if we find something in the train_data_directory , check if this is a directory and if it is one, 
    # we will add to our list"directories"
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    #Inicialize two list, labels and images
    labels = []
    images = []
    #Loop through the subdirectories
    #for each subdirectories
    for d in directories:
        #each subdirectory represents a label, then we save the label_directory
        label_directory = os.path.join(data_directory, d)
        #we save each names of files in subdirectorie in a list
        file_names = [os.path.join(label_directory, f) 
                      #for each "something"
                      for f in os.listdir(label_directory) 
                      #check if is a file with extension .ppm (a image)
                      if f.endswith(".ppm")]
        #we loop through the names of files
        for f in file_names:
#             print(file_names)
#             print(f)
            #append image with label image
            images.append(skimage.data.imread(f))
            #d is a name of subdirectory and each name of subdirectory is a label
            labels.append(int(d))
    return images, labels

#Stablished ROOT_PATH to images folders.
ROOT_PATH = "C:/Users/Al_ko/Tutorial DataCamp"
#with os.path.join we can add the specific paths to our ROOT_PATH
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")
#Load images and labels from train_data_directory
images, labels = load_data(train_data_directory)

# +
#Note that the images and labels variables are lists, we need use np.array() 
#to conver list into array. Then we can use ndim and size method on array variable.

# Print the `images` dimensions
print("Number images dimensions:",np.array(images).ndim )

# Print the number of `images`'s elements
print("Number of images elements:",np.array(images).size)

# Print the first instance of `images`
#Note that the images[0] that you printed out is, in fact, one single image that is represented by arrays in arrays! 
#This might seem counterintuitive at first, but it’s something that you’ll get used to as you go further 
#into working with images in machine learning or deep learning applications.
#tensor range 3?

#Print the image [0]
print("Show images[0]:")
print(images[300])
#Print flags from the image [0]
print("show flags from images[0]")
print(images[0].flags)
# print num items from image[0]?
print("Show size of images[0]", images[0].itemsize)
# print bytes from the image [0]
print("Bytes from images[0]")
print(images[0].nbytes)

#Mostrar valor RGB de un pixel
arrayImagenes = np.array(images)
pixel = arrayImagenes[300][0][0]
print(pixel)

# +
# Print the `labels` dimensions
print("Number of labels dimensions:",np.array(labels).ndim)

# Print the number of `labels`'s elements
print("Number of labels elements:",np.array(labels).size)

# Count the number of labels
print("Count number of different labels:",len(set(labels)))

# +
# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)

# Show the plot
plt.show()

#You clearly see that not all types of traffic signs are equally represented in the dataset. 
#This is something that you’ll deal with later when you’re manipulating the data before you start modeling your neural network.

#At first sight, you see that there are labels that are more heavily present in the dataset than others: 
#the labels 22, 32, 38, and 61 definitely jump out. At this point, it’s nice to keep this in mind, 
#but you’ll definitely go further into this in the next section!
# -

#To plot only a img with plt.imgshow(image)
plt.imshow(images[3650])

# +
#Now, we are going to make a list with 4 random numbers. These will be used to select traffic signs from 
#the images array that you have just inspected in the previous section. 
#In this case, we use 300, 2250, 3650 and 4000 to plot 4 random images

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]
#To plot only a img with plt.imgshow(image)
# plt.imshow(images[400])

# Fill out the subplots with the random images that we defined 
for i in range(len(traffic_signs)):
#   To generate area to plot in each iterations
    plt.subplot(1, 4, i+1)
    #To show axis in each image(option on or off)
    plt.axis('on')
    #To plot image as image using imshow(image)
    plt.imshow(images[traffic_signs[i]])
#   the amount of width reserved for space between subplots,
#   expressed as a fraction of the average axis width
    plt.subplots_adjust(wspace=0.5)

# -

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))

# +
#Now that we are see 4 random images, go plot 1 image of each label class.

# Get the unique labels 
unique_labels = set(labels)

#To rescale a figure in plot graphs. In this case, we rescale each image to a 15x15px.
# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    #Define the space to plots, in this case we are plot 62 images(we need a grid of 64), 
    #and in each iter in a grid of 8x8 we plot in position i
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes on each subplot
    plt.axis('off')
    # Add a title to each subplot
    #{0} label name
    #{1} count of labels of that class.
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
plt.show()

#we can see that we have more items of some class, for example label 22, label 32, similar that we see in the histogram above.
#Ademas, podriamos intentar buscar alguna relación ya, por ejemplo, pensar si hay alguna relación con que las clases que
#mas estan representadas son de tipo prohibicion, y aunque esta hipotesis es valida para las clases 22 y 32, e incluso con la 38, 
#esto no se cumple para la clase 61.
#La unica conclusion a priori que se puede sacar es que mas de la mitad del dataset son señales de prohibicion.

# +
#Hasta que hemos visto el dataset, podemos resumir lo siguiente:
#El tamaño o dimensiones de las imagenes es distinto.
#Tenemos 62 clases(empezando en 0 a 61)
# La distribucion de las clases es desbalanceada, y no hay conexion entre esas clases.
# Ahora que tenemos claro lo que tenemos podemos empezar a manipular los datos, de manera que los dejemos preparados
#para alimentar nuestra red neuronal.

# Import the `transform` module from `skimage`
#from skimage import transform 

#Esta declarando la lista resultante de aplicar la funcion transform.resize(imagen,(dimensiones del reescalado) 
# variable = [] => lista vacia
#para cada elemento de la lista images)
#Ahora images 28
# Rescale the images in the `images` array
#NOTA: Al reescalar el tamaño, tambien se normalizan los valores de lo pixeles RBG.
images28 = [transform.resize(image, (28, 28)) for image in images]
# images28 = np.array(images32)

#shape me dice alto, ancho y el numero de elementos que tiene cada componente(en este caso 3, componentes RGB)
print("shape images28[1]:",images28[1].shape)
print("shape images[1]:",images[1].shape)
#Si convertimos images28 a array, e imprimimos su shape vemos que tenemos:
#numero de fotos, dimensiones de cada foto(alto y ancho) y el valor de cada componente.
array28 = np.array(images28)
print("Array(images28).shape:",array28.shape)
array = np.array(images)
#En este caso no podra mostrarnos bien el shape porque cada imagen tiene dimensiones distintas
print("Array(images).shape:",array.shape)
#Ahora las imagenes tendran 784 dimensiones(28x28)
print("Array(images[1].shape)",array[1].shape)




# -

# Fill out the subplots with the random images and add shape, min and max values
#this plot is on images28
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images28[traffic_signs[i]].shape, 
                                                  images28[traffic_signs[i]].min(), 
                                                  images28[traffic_signs[i]].max()))

# We can see as the value of RGB pixel change with rescale
print(np.array(images28[0]))

# +
#In the datacamp tutorial said that the color in pictures matters less when you are trygint to answer a classification question?
#Then we are going to convert the images in greyscale but too we will do without convert and test result.
# Import `rgb2gray` from `skimage.color`
# from skimage.color import rgb2gray

#function: rgb2gray(array)
# Convert `images28` to grayscale
images28grey = rgb2gray(np.array(images28))
# images28 = rgb2gray(np.array(images28))

# +
#Plot images
#Plot traffic_signs in raw
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    
# Show the plot
plt.show()

#Plot images rescaled 28x28
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    
# Show the plot
plt.show()
#Plot traffic_signs on greyscale images
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    #we have to specify the color map or cmap and set it to "gray" to plot the images in grayscale. 
    #That is because imshow() by default uses a heatmap-like color map.
    #Read more about this: https://stackoverflow.com/questions/39805697/skimage-why-does-rgb2gray-from-skimage-color-result-in-a-colored-image
    plt.imshow(images28grey[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    
# Show the plot
plt.show()

#if we dont set cmap to gray
#Plot traffic_signs on greyscale images
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28grey[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    
# Show the plot
plt.show()
#Funcion para plotear, dado un array, indices a plotear y colormap
def plotear(toPlot, indices, colormap):
    for i in range(len(indices)):
        #Maximo mostraremos 8
        if(len(indices) > 8):
            n = 8
        else:
            n = len(indices)
        #Ploteamos de 1 a N en la posicion i+1  
        plt.subplot(1,n, i+1)
        #Eliminamos ejes sobre cada subplot
        plt.axis('off')
        #para poder pintar con escalas de grises
        if(colormap == "grey"):
            plt.imshow(toPlot[indices[i]], cmap="grey")
        else:
            plt.imshow(toPlot[indices[i]])
        #Añadir margen
        plt.subplots_adjust(wspace=0.5)
        
    #Show the plot
    plt.show()

plotear(images,traffic_signs,"none")


# +
#Plot traffic_signs on greyscale images
plt.imshow(arrayImagenes[300][25])
    
# Show the plot
plt.show()
#Mostrar un unico pixel
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('on')
    plt.imshow([[[0,0,0]]])
    plt.subplots_adjust(wspace=0.5)

plt.show()
    
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)
# -
############################ BELGIAN TRAFFIC SIGNS ##########################################
####################### DEEP LEARNING WITH TENSORFLOW #######################################
print("This is a tutorial of neural network with TrafficSigns from DataCamp in tensorflow:")
print("https://www.datacamp.com/community/tutorials/tensorflow-tutorial")
print("Go start to define tensors and our neural network for TrafficSigns Tuturial")

# +
#To start, we should know...
# we can initialize the Graph with the help of Graph(). You use this function to define the computation. 
# Note that with the Graph, you don’t compute anything, because it doesn’t hold any values. It just defines the 
# operations that you want to be running later.
# In this case, you set up a default context with the help of as_default(), which returns a context manager that
# makes this specific Graph the default graph. You use this method if you want to create multiple graphs in the 
# same process: with this function, you have a global default graph to which all operations will be added if you 
# don’t explicitly create a new graph.
import tensorflow as tf 

#First, we define placeholders for inputs and labels because you won’t put in the “real” data yet. Remember that 
#placeholders are values that are unassigned and that will be initialized by the session when you run it. So when 
#you finally run the session, these placeholders will get the values of your dataset that you pass in the run() function!
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name="x")
y = tf.placeholder(dtype = tf.int32, shape = [None], name = "y")

#You first start by flattening the input with the help of the flatten() function, which will give you 
#an array of shape [None, 784] instead of the [None, 28, 28], which is the shape of your grayscale images.
# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

#we construct a fully connected layer that generates logits of size [None, 62]. - Porque None,62?
#Logits is the function operates on the unscaled output of previous layers, and that uses the 
#relative scale to understand the units is linear.
#Cuando computamos la entrada por el bias(la neurona), el resultado se desnormaliza?, logits 
#usa la escala relativa para que sean unidades lineales? normalizadas?
#Otra cosa interesante a destacar, es que hace uso de la funcion relu para la activacion de las neuronas
#tf.nn es un modulo de tensorflow para envolver operaciones en redes neuronales(por ejemplo la operacion relu)
#De manera que logits ahora es una capa de las imagenes flateadas a 62 neuronas, aplicando la funcion relu de activacion(creo)
#logits es 62, porque tenemos 62 clases de señales, y obtendremos valores de regresion para cada una de las clases
#sobre los cuales calculara posteriormente la perdida que tienen dichos valores.
# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

#Para la funcion de perdida

#The choice for a loss function depends on the task that you have at hand: in this case, you make use of
#the function sparse_softmax_cross_entropy_with_logits()
#This computes sparse softmax cross entropy between logits and labels. In other words, it measures the probability 
#error in discrete classification tasks in which the classes are mutually exclusive. This means that each entry is 
#in exactly one class. Here, a traffic sign can only have one single label. Remember that, while regression is 
#used to predict continuous values, classification is used to predict discrete values or classes of data points. 
#You wrap this function with reduce_mean(), which computes the mean of elements across dimensions of a tensor.

#En otras palabras, con el reduce_mean calculamos(declaramos la operacion) para obtener un unico valor de perdida
#Y softmax es un clasificador de regresion, es decir que te da un conjunto de valores entre 0 y 1 para la pertenencia de la clase
#Esto en clasificacion o en clasificaciones que son mutuamente exclusivas(no pueden pertenecer a dos clases), nos daria fallo
#para la optimización, porque no tenemos un unico valor, si no la probabilidad de pertenecer a cada una de las clases, por
#lo que entiendo que con esta funcion lo que hace es decidirse por un unico valor de pertenencia a clase
#que es el que utiliza para la definicion de la funcion de perdida
#
#Tal y como se introdujo antes, tf.nn es un modulo que envuelve operaciones sobre redes neuronales y que en nuestro caso
#aparte de tener la funcion relu, tiene la funcion sparse_softmax_cross_entropy_with_logits
#aunque segun documentacion de tf esta deprecated utilizandose "tf.nn.softmax_cross_entropy_with_logits_v2"
#See: https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
#En resumen, aqui lo que hacemos es definir la funcion de perdida, es decir, que itere y vea que diferencia hay entre el 
#valor obtenido, 
#Calculamos pues la perdida de los valores que teniamos en logits con las etiquetas que deberian ser
# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

#We also want to define a training optimizer; Some of the most popular optimization algorithms used are 
#the Stochastic Gradient Descent (SGD), ADAM and RMSprop(Root Mean Squared propagation?). Depending on whichever algorithm you choose, 
#you’ll need to tune certain parameters, such as learning rate or momentum. In this case, you pick the ADAM optimizer, 
#for which you define the learning rate at 0.001.
#un optimizador que va a minimizar la perdida
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#Segun la documentacion de tensorflow
#Returns the index with the largest value across axes of a tensor
#En esencia aqui, es para que cuando le pasemos una "tupla", imagen o lo que sea, nos va a dar un valor de regresion con 
#el valor de pertenencia a cada una de las clases, con argmax lo que hacemos es aplanar dicho valor, es decir
#quedarnos con el mayor valor o the largest value accross axes of a tensor ;) 
# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

#tf.cast lo que hace es cambiar el tipo de un tensor a otro, en este caso, el tipo de tensor correct_pred a tf.float32
accuracy_test = tf.reduce_mean(correct_pred)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("x: ", x)
print("y: ", y)
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("Train_op: ", train_op)
print("Correct_pred without cast:", correct_pred)
print("Accuracy_test: ", accuracy_test)
print("Correct_pred with cast:", tf.cast(correct_pred, tf.float32))
print("Accuracy: ", accuracy)
#Lo que ocurre es que correct_pred es de tipo entero, y si queremos definir la accuracy del modelo en base a correct_pred
#por defecto el tensor sera de tipo entero(igual que correct_pred), pero que el valor de accuracy sea entero no sirve de nada
#necesitas un flotante
#Podemos ver los tensores en los anteriores print

# +
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')


# +
tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)

# +
# Import `matplotlib`
import matplotlib.pyplot as plt
import random

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()
# -


