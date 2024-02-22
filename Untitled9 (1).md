```python
import numpy as np
import pandas as pd
from sklearn import preprocessing 
import tensorflow as tf
```


```python
df=np.loadtxt(r'C:\Users\keyvans\OneDrive - Sealegs Ltd\Desktop\Jupyter notebook\Audiobooks_data.csv', delimiter=',')
```


```python

```


```python
df.shape[0]
```




    14084




```python
unscalled_input_all=df[:,1:-1]
target_all=df[:,-1:]
```


```python
num_one=int(np.sum(target_all.shape[0]))
indice_remove=[]
num_zero=0
for i in range(target_all.shape[0]):
    if target_all[i]==0:
        num_zero+=1
        if num_zero>num_one:
            indec_append(i)
            
    
```


```python
unscalled_inputs=np.delete(unscalled_input_all,indice_remove, axis=0)
targets_all2=np.delete(target_all,indice_remove, axis=0)
```


```python
scalled_input=preprocessing.scale(unscalled_inputs)
targets_all2=np.delete(target_all,indice_remove, axis=0)
```


```python
shuffled_indice=np.arange(scalled_input.shape[0])
np.random.shuffle(shuffled_indice)

shuffel_input=scalled_input[shuffled_indice]
shuffled_target=targets_all2[shuffled_indice]
```


```python
sample_count=int(shuffel_input.shape[0])
inputs_num=int(.8 * sample_count)
valid_num=int(.1 * sample_count)
test_num=sample_count-inputs_num-valid_num
```


```python
inputs=shuffel_input[:inputs_num]
targets=shuffled_target[:inputs_num]

valid_input=shuffel_input[inputs_num:inputs_num+valid_num]
valid_target=shuffled_target[inputs_num:inputs_num+valid_num]

test_input=shuffel_input[inputs_num+valid_num:]
test_targets=shuffled_target[inputs_num+valid_num:]

```


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Determine the number of input features
input_shape = inputs.shape[1]

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),  # First hidden layer with 64 neurons
    Dense(32, activation='relu'),  # Second hidden layer with 32 neurons
    Dense(1, activation='sigmoid')  # Output layer, assuming binary classification
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
              metrics=['accuracy'])

# Train the model
model.fit(inputs, targets, epochs=100, batch_size=100, validation_data=(valid_input, valid_target))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_input, test_targets)
print("\nTest accuracy:", test_accuracy)

```

    WARNING:tensorflow:From C:\Users\keyvans\AppData\Roaming\Python\Python311\site-packages\keras\src\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From C:\Users\keyvans\AppData\Roaming\Python\Python311\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    Epoch 1/100
    WARNING:tensorflow:From C:\Users\keyvans\AppData\Roaming\Python\Python311\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
    
    WARNING:tensorflow:From C:\Users\keyvans\AppData\Roaming\Python\Python311\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    113/113 [==============================] - 2s 5ms/step - loss: 0.4311 - accuracy: 0.8330 - val_loss: 0.2889 - val_accuracy: 0.9055
    Epoch 2/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2910 - accuracy: 0.8958 - val_loss: 0.2532 - val_accuracy: 0.9162
    Epoch 3/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2657 - accuracy: 0.9002 - val_loss: 0.2426 - val_accuracy: 0.9126
    Epoch 4/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2547 - accuracy: 0.9027 - val_loss: 0.2379 - val_accuracy: 0.9070
    Epoch 5/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2499 - accuracy: 0.9036 - val_loss: 0.2289 - val_accuracy: 0.9197
    Epoch 6/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2447 - accuracy: 0.9053 - val_loss: 0.2262 - val_accuracy: 0.9197
    Epoch 7/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2429 - accuracy: 0.9067 - val_loss: 0.2280 - val_accuracy: 0.9169
    Epoch 8/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2415 - accuracy: 0.9063 - val_loss: 0.2226 - val_accuracy: 0.9212
    Epoch 9/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2386 - accuracy: 0.9088 - val_loss: 0.2308 - val_accuracy: 0.9141
    Epoch 10/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2369 - accuracy: 0.9079 - val_loss: 0.2202 - val_accuracy: 0.9197
    Epoch 11/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2340 - accuracy: 0.9081 - val_loss: 0.2193 - val_accuracy: 0.9190
    Epoch 12/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2322 - accuracy: 0.9087 - val_loss: 0.2217 - val_accuracy: 0.9134
    Epoch 13/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2332 - accuracy: 0.9088 - val_loss: 0.2164 - val_accuracy: 0.9176
    Epoch 14/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2313 - accuracy: 0.9085 - val_loss: 0.2237 - val_accuracy: 0.9141
    Epoch 15/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2307 - accuracy: 0.9104 - val_loss: 0.2181 - val_accuracy: 0.9176
    Epoch 16/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2298 - accuracy: 0.9094 - val_loss: 0.2180 - val_accuracy: 0.9190
    Epoch 17/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2294 - accuracy: 0.9102 - val_loss: 0.2182 - val_accuracy: 0.9119
    Epoch 18/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2284 - accuracy: 0.9101 - val_loss: 0.2159 - val_accuracy: 0.9190
    Epoch 19/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2271 - accuracy: 0.9098 - val_loss: 0.2146 - val_accuracy: 0.9197
    Epoch 20/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2292 - accuracy: 0.9093 - val_loss: 0.2165 - val_accuracy: 0.9183
    Epoch 21/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2272 - accuracy: 0.9107 - val_loss: 0.2141 - val_accuracy: 0.9212
    Epoch 22/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2267 - accuracy: 0.9102 - val_loss: 0.2244 - val_accuracy: 0.9105
    Epoch 23/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2271 - accuracy: 0.9085 - val_loss: 0.2139 - val_accuracy: 0.9212
    Epoch 24/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2255 - accuracy: 0.9094 - val_loss: 0.2171 - val_accuracy: 0.9183
    Epoch 25/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2248 - accuracy: 0.9120 - val_loss: 0.2184 - val_accuracy: 0.9112
    Epoch 26/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2243 - accuracy: 0.9106 - val_loss: 0.2150 - val_accuracy: 0.9169
    Epoch 27/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2263 - accuracy: 0.9108 - val_loss: 0.2151 - val_accuracy: 0.9183
    Epoch 28/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2254 - accuracy: 0.9106 - val_loss: 0.2199 - val_accuracy: 0.9205
    Epoch 29/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2249 - accuracy: 0.9101 - val_loss: 0.2120 - val_accuracy: 0.9212
    Epoch 30/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2259 - accuracy: 0.9097 - val_loss: 0.2186 - val_accuracy: 0.9183
    Epoch 31/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2240 - accuracy: 0.9105 - val_loss: 0.2138 - val_accuracy: 0.9197
    Epoch 32/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2244 - accuracy: 0.9104 - val_loss: 0.2179 - val_accuracy: 0.9141
    Epoch 33/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2252 - accuracy: 0.9089 - val_loss: 0.2116 - val_accuracy: 0.9219
    Epoch 34/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2236 - accuracy: 0.9110 - val_loss: 0.2128 - val_accuracy: 0.9162
    Epoch 35/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2241 - accuracy: 0.9121 - val_loss: 0.2136 - val_accuracy: 0.9205
    Epoch 36/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2250 - accuracy: 0.9101 - val_loss: 0.2129 - val_accuracy: 0.9183
    Epoch 37/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2241 - accuracy: 0.9104 - val_loss: 0.2190 - val_accuracy: 0.9134
    Epoch 38/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2233 - accuracy: 0.9117 - val_loss: 0.2121 - val_accuracy: 0.9219
    Epoch 39/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2242 - accuracy: 0.9101 - val_loss: 0.2119 - val_accuracy: 0.9197
    Epoch 40/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2241 - accuracy: 0.9113 - val_loss: 0.2093 - val_accuracy: 0.9219
    Epoch 41/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2240 - accuracy: 0.9113 - val_loss: 0.2171 - val_accuracy: 0.9091
    Epoch 42/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2224 - accuracy: 0.9117 - val_loss: 0.2329 - val_accuracy: 0.9134
    Epoch 43/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2249 - accuracy: 0.9109 - val_loss: 0.2129 - val_accuracy: 0.9190
    Epoch 44/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2239 - accuracy: 0.9123 - val_loss: 0.2119 - val_accuracy: 0.9226
    Epoch 45/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2231 - accuracy: 0.9122 - val_loss: 0.2123 - val_accuracy: 0.9233
    Epoch 46/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2240 - accuracy: 0.9112 - val_loss: 0.2124 - val_accuracy: 0.9190
    Epoch 47/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2221 - accuracy: 0.9119 - val_loss: 0.2113 - val_accuracy: 0.9226
    Epoch 48/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2232 - accuracy: 0.9094 - val_loss: 0.2132 - val_accuracy: 0.9190
    Epoch 49/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2230 - accuracy: 0.9119 - val_loss: 0.2092 - val_accuracy: 0.9219
    Epoch 50/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2232 - accuracy: 0.9114 - val_loss: 0.2156 - val_accuracy: 0.9190
    Epoch 51/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2234 - accuracy: 0.9091 - val_loss: 0.2126 - val_accuracy: 0.9226
    Epoch 52/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2226 - accuracy: 0.9122 - val_loss: 0.2106 - val_accuracy: 0.9197
    Epoch 53/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2230 - accuracy: 0.9119 - val_loss: 0.2189 - val_accuracy: 0.9119
    Epoch 54/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2231 - accuracy: 0.9116 - val_loss: 0.2216 - val_accuracy: 0.9098
    Epoch 55/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2227 - accuracy: 0.9111 - val_loss: 0.2179 - val_accuracy: 0.9155
    Epoch 56/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2218 - accuracy: 0.9116 - val_loss: 0.2176 - val_accuracy: 0.9119
    Epoch 57/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2216 - accuracy: 0.9128 - val_loss: 0.2122 - val_accuracy: 0.9190
    Epoch 58/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2223 - accuracy: 0.9118 - val_loss: 0.2090 - val_accuracy: 0.9219
    Epoch 59/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2214 - accuracy: 0.9116 - val_loss: 0.2175 - val_accuracy: 0.9062
    Epoch 60/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2226 - accuracy: 0.9114 - val_loss: 0.2120 - val_accuracy: 0.9197
    Epoch 61/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2203 - accuracy: 0.9129 - val_loss: 0.2111 - val_accuracy: 0.9212
    Epoch 62/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2230 - accuracy: 0.9113 - val_loss: 0.2099 - val_accuracy: 0.9226
    Epoch 63/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2220 - accuracy: 0.9125 - val_loss: 0.2106 - val_accuracy: 0.9226
    Epoch 64/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2211 - accuracy: 0.9114 - val_loss: 0.2108 - val_accuracy: 0.9183
    Epoch 65/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2213 - accuracy: 0.9128 - val_loss: 0.2242 - val_accuracy: 0.9169
    Epoch 66/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2214 - accuracy: 0.9115 - val_loss: 0.2111 - val_accuracy: 0.9219
    Epoch 67/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2216 - accuracy: 0.9110 - val_loss: 0.2163 - val_accuracy: 0.9148
    Epoch 68/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2218 - accuracy: 0.9125 - val_loss: 0.2103 - val_accuracy: 0.9212
    Epoch 69/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2207 - accuracy: 0.9137 - val_loss: 0.2119 - val_accuracy: 0.9190
    Epoch 70/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2224 - accuracy: 0.9123 - val_loss: 0.2113 - val_accuracy: 0.9205
    Epoch 71/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2236 - accuracy: 0.9112 - val_loss: 0.2161 - val_accuracy: 0.9098
    Epoch 72/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2189 - accuracy: 0.9121 - val_loss: 0.2117 - val_accuracy: 0.9219
    Epoch 73/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2215 - accuracy: 0.9112 - val_loss: 0.2126 - val_accuracy: 0.9183
    Epoch 74/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2203 - accuracy: 0.9118 - val_loss: 0.2128 - val_accuracy: 0.9155
    Epoch 75/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2215 - accuracy: 0.9117 - val_loss: 0.2121 - val_accuracy: 0.9205
    Epoch 76/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2200 - accuracy: 0.9125 - val_loss: 0.2155 - val_accuracy: 0.9119
    Epoch 77/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2208 - accuracy: 0.9112 - val_loss: 0.2116 - val_accuracy: 0.9183
    Epoch 78/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2201 - accuracy: 0.9124 - val_loss: 0.2179 - val_accuracy: 0.9197
    Epoch 79/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2226 - accuracy: 0.9114 - val_loss: 0.2237 - val_accuracy: 0.9070
    Epoch 80/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2204 - accuracy: 0.9128 - val_loss: 0.2125 - val_accuracy: 0.9190
    Epoch 81/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2201 - accuracy: 0.9128 - val_loss: 0.2101 - val_accuracy: 0.9219
    Epoch 82/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2218 - accuracy: 0.9115 - val_loss: 0.2095 - val_accuracy: 0.9226
    Epoch 83/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2204 - accuracy: 0.9126 - val_loss: 0.2115 - val_accuracy: 0.9212
    Epoch 84/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2203 - accuracy: 0.9129 - val_loss: 0.2096 - val_accuracy: 0.9176
    Epoch 85/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2197 - accuracy: 0.9126 - val_loss: 0.2114 - val_accuracy: 0.9162
    Epoch 86/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2195 - accuracy: 0.9126 - val_loss: 0.2125 - val_accuracy: 0.9183
    Epoch 87/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2202 - accuracy: 0.9116 - val_loss: 0.2108 - val_accuracy: 0.9219
    Epoch 88/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2190 - accuracy: 0.9136 - val_loss: 0.2109 - val_accuracy: 0.9197
    Epoch 89/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2205 - accuracy: 0.9116 - val_loss: 0.2109 - val_accuracy: 0.9212
    Epoch 90/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2187 - accuracy: 0.9120 - val_loss: 0.2098 - val_accuracy: 0.9212
    Epoch 91/100
    113/113 [==============================] - 1s 5ms/step - loss: 0.2195 - accuracy: 0.9116 - val_loss: 0.2114 - val_accuracy: 0.9183
    Epoch 92/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2203 - accuracy: 0.9124 - val_loss: 0.2117 - val_accuracy: 0.9190
    Epoch 93/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2228 - accuracy: 0.9125 - val_loss: 0.2093 - val_accuracy: 0.9226
    Epoch 94/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2205 - accuracy: 0.9120 - val_loss: 0.2153 - val_accuracy: 0.9112
    Epoch 95/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2183 - accuracy: 0.9136 - val_loss: 0.2082 - val_accuracy: 0.9226
    Epoch 96/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2204 - accuracy: 0.9129 - val_loss: 0.2142 - val_accuracy: 0.9205
    Epoch 97/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2204 - accuracy: 0.9121 - val_loss: 0.2084 - val_accuracy: 0.9212
    Epoch 98/100
    113/113 [==============================] - 0s 4ms/step - loss: 0.2185 - accuracy: 0.9134 - val_loss: 0.2094 - val_accuracy: 0.9233
    Epoch 99/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2184 - accuracy: 0.9129 - val_loss: 0.2132 - val_accuracy: 0.9212
    Epoch 100/100
    113/113 [==============================] - 0s 3ms/step - loss: 0.2190 - accuracy: 0.9142 - val_loss: 0.2143 - val_accuracy: 0.9155
    45/45 [==============================] - 0s 2ms/step - loss: 0.2176 - accuracy: 0.9120
    
    Test accuracy: 0.9119943380355835
    


```python

```
