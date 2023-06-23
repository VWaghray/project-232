from numpy import loadtxt
import numpy
from keras.models import Sequential
from keras.layers import Dense

data_set = loadtxt('pokemon.csv', delimiter=',')

x=data_set[:,1:7]
y=data_set[:,8]

model=Sequential()
model.add(Dense(12, input_dim=6,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x,y,epochs=250,batch_size=100)
predictions=numpy.argmax(model.predict(x), axis=-1)
for i in range(785,800):
	print(f'{x[i].tolist()} => {predictions[i]} expected {y[i]}')
