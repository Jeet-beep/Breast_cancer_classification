import pickle
import numpy as np
model_name="breast_svm(1).pkl"
with open(model_name,'rb') as file:
  a=pickle.load(file)
print(a)
input_data=(914580,12.47,17.31,	80.45,	480.1,	0.08928,	0.0763,	0.03609,	0.02369,	0.1526,	0.06046,	0.1532,	0.781,	1.253,	11.91,	0.003796,	0.01371,	0.01346,	0.007096,	0.01536,	0.001541,	14.06,	24.34,	92.82,	607.3,	0.1276,	0.2506,	0.2028,	0.1053,	0.3035,	0.07661)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data) # converting this list into numpy array

# reshape the numpy array as we are predicting for one instance

input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)

predictions = a.predict(input_data_reshaped)
print(predictions)
if predictions[0] == 1:
    print("M")
else:
    print("B")