import torch
from transformers import XLNetForSequenceClassification
from transformers import logging
logging.set_verbosity_error()
PATH = "./xlnet_model.bin"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 2)
model.load_state_dict(torch.load(PATH,  map_location=torch.device('cpu')))
model.to(device)
