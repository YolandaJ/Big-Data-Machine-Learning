import pandas as pd

# read file from given path
def read(path):
    data = pd.read_csv(path, sep=",", header=None)
    return data
    # TEST OK

# removing record having missing or incomplete features    
def removeMissing(data):
    data.dropna() # drop rows whose any value is NaN
    nrow, ncol = data.shape[0], data.shape[1]
    for i in range(ncol):
        data = data[~data[i].isin(["None", "none", ""])]
    return data
    # TEST OK

# If the value is numeric, standardized, which means each value subtracting the mean and dividing by the standard deviation.
# If the value is categorical or nominal, it needs to be converted to numerical values.
def normalize(data):
    nrow, ncol = data.shape[0], data.shape[1]
    for i in range(ncol):
        if (data[i].dtype == 'object'):
            # Encode each column as an enumerated type or categorical variable
            data[i] = pd.factorize(data[i])[0]
        else :
            mean = data[i].mean()
            std = data[i].std()
            data[i] = data[i].apply(lambda x : (x - mean) / std)
    return data
  
# output the processed data to a given path
def output(data, outpath):
    data.to_csv(outpath, index = False)
    return       
            
# ###########################
path = input("Please enter your raw data path: ")
outpath = input("Please give your output path: ")
data = read(path)
nomissing = removeMissing(data)
normal = normalize(nomissing)
print("You have got the processed data in your current directory. Go and have a look.")
output(normal, outpath)