import gc
import torch
import pickle

def save_model(model, path):
  with open("models/"+path, 'wb') as f:
    pickle.dump(model, f)

def load_model(path):
  with open("models/"+f'{path}.pkl', 'rb') as f:
    model = pickle.load(f)
  return model

def get_batch(input, mult=1):
    """ Batch size is 10 x mult """
    xx = []
    yy = []
    # idx = np.random.randint(MSRpar.shape[0], size=1 * mult);
    # xx.extend(MSRpar[1][idx]); yy.extend(MSRpar[2][idx])
    for i in range(1 * mult):
      # xx.append(input[0].values[i]); yy.append(input[1].values[i]) # 
      xx.append(input[i][0]); yy.append(input[i][1]) 
    return xx, yy

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
