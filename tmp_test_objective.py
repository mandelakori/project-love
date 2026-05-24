import pandas as pd
from train import load_training_data, objective as cpu_objective
from train_gpu import objective as gpu_objective, detect_gpu
from sklearn.model_selection import train_test_split
X,y = load_training_data()
X_train = X.head(2000).drop(columns=['year'])
y_train = y[:2000]
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
dummy = type('T',(object,),{
    'suggest_int': lambda *args, **kwargs: kwargs.get('low', 3) if args[0] == 'max_depth' else 50,
    'suggest_float': lambda *args, **kwargs: 0.1
})
ll = cpu_objective(dummy, X_train, y_train, X_val, y_val)
print('CPU objective ok', ll)
method, dev = detect_gpu()
ll2 = gpu_objective(dummy, X_train, y_train, X_val, y_val, method, dev)
print('GPU objective ok', ll2)
