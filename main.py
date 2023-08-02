import keras as k
import numpy as np
import matplotlib.pyplot as plt

x = np.array([-50, -25, 0, 8, 15, 22, 38])
y = np.array([-58, -13, 32, 46.4, 59, 71.6, 100.4])

model = k.Sequential()
model.add(k.layers.Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(loss='mean_squared_error', optimizer=k.optimizers.Adam(0.1))

fit_results = model.fit(x=x, y=y, epochs=1000, verbose=0)
predicted = model.predict([6000])
print(predicted)

plt.plot(fit_results.history['loss'])
plt.grid(True)
plt.show()
