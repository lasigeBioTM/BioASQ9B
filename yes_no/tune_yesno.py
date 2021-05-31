import argparse
import json
import numpy as np
import pandas as pd
import pickle
import torch
import random
import jsonlines
from pathlib import Path
from torch import nn
from barbar import Bar


def main():

	print('Start')
	parser = argparse.ArgumentParser()

	# Add the arguments to the parser
	parser.add_argument("--model_name", required= True)
	parser.add_argument("--checkpoint_input_path", required= False)
	parser.add_argument("--checkpoint_output_path", required= True)
	parser.add_argument("--bioasq_path", required= True)
	parser.add_argument("--seed", default = 1995)
	parser.add_argument("--learning_rate", default = 5e-5, type = float)
	parser.add_argument("--batch_size", default = 16, type = int)
	parser.add_argument("--epochs", default = 3, type=int)
	
	parser.add_argument('--mid_layer', dest='mid_layer', action='store_true')
	parser.add_argument('--no-mid_layer', dest='mid_layer', action='store_false')
	parser.set_defaults(mid_layer=True)

	parser.add_argument('--balance', dest='balance', action='store_true')
	parser.add_argument('--no-balance', dest='balance', action='store_false')
	parser.set_defaults(balance=True)

	parser.add_argument("--mid_layer_size", default = 256, type = int)







	args = vars(parser.parse_args())

	print(args['mid_layer'])

	random.seed(args['seed'])
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



	with open(args['bioasq_path'], 'rb') as f:
		bio_yn_raw = json.load(f)['questions']
	bio_yn = [question for question in bio_yn_raw if question['type'] == 'yesno']
	bio_yn_questions = [question['body'] for question in bio_yn]
	bio_yn_ids = [question['id'] for question in bio_yn]
	bio_yn_answers = [question['exact_answer'] for question in bio_yn]
	bio_snippets = {question['id'] : [snippet['text'] 
	                                  for snippet in question['snippets']] 
	                for question in bio_yn}

	ids = []
	snippets = []
	for key, value in bio_snippets.items():
	    for snippet in value:
	        ids.append(key)
	        snippets.append(snippet)

	snippets_df = pd.DataFrame({'id': ids,'snippet': snippets})
	questions_df = pd.DataFrame({'id': bio_yn_ids, 
	                             'question': bio_yn_questions,
	                            'label': bio_yn_answers})
	bio_yn_df = pd.merge(snippets_df,questions_df, how = 'left', on = 'id')

	bio_yn_df = bio_yn_df.sample(32)
	no_size = bio_yn_df[bio_yn_df.label == 'no'].shape[0]
	yes_index = bio_yn_df[bio_yn_df.label == 'yes'].index
	random_index = np.random.choice(yes_index, no_size, replace=False)
	yes_sample = bio_yn_df.loc[random_index]
	bio_yn_balanced = pd.concat([yes_sample,bio_yn_df[bio_yn_df.label == 'no']])

	bio_yn_balanced = bio_yn_balanced.sample(frac=1)

	if args['balance']:
		train_a = list(bio_yn_balanced.question)
		train_b = list(bio_yn_balanced.snippet)
		train_labels = [int(answer == 'yes') for answer in bio_yn_balanced.label]
	else:
		train_a = list(bio_yn_df.question)
		train_b = list(bio_yn_df.snippet)
		train_labels = [int(answer == 'yes') for answer in bio_yn_df.label]



	from transformers import BertTokenizer
	# Load the BERT tokenizer.
	tokenizer = BertTokenizer.from_pretrained(args['model_name'], 
	                                          do_lower_case=True)


	# In[39]:


	train_tokens = tokenizer(train_a,train_b, 
	                       add_special_tokens=True,
	                       max_length=500,
	                       truncation=True, padding=True)
	train_tokens['labels'] = train_labels


	# In[40]:


	from torch.utils.data import Dataset, DataLoader

	class MnliDataset(Dataset):
	    def __init__(self, encodings):
	        self.encodings = encodings

	    def __getitem__(self, idx):
	        #print(self.encodings['start_positions'][idx])
	        #{key: torch.tensor(val[idx], dtype = torch.long) for key, val in self.encodings.items()}
	        return {'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype = torch.long),
	                'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype = torch.long),
	                'token_type_ids': torch.tensor(self.encodings['token_type_ids'][idx], dtype = torch.long),
	                'labels': torch.tensor(self.encodings['labels'][idx], dtype = torch.long)
	               }

	    def __len__(self):
	        return len(self.encodings.input_ids)

	train_dataset = MnliDataset(train_tokens)


	# In[5]:

	# In[4]:


	from transformers import BertForSequenceClassification

	model = BertForSequenceClassification.from_pretrained(args['model_name'], 
		num_labels = 3)
	checkpoint = torch.load(args['checkpoint_input_path'],map_location=device)
	model.load_state_dict({key.replace('module.',''): value for key,value in checkpoint.items()})
	# freeze all the parameters
	#for param in model.parameters():
	#    param.requires_grad = False
	# In[73]:


	class BERT_Arch(nn.Module):

	    def __init__(self, model):
	      
	        super(BERT_Arch, self).__init__()

	        self.model = model
	        
	        # dropout layer
	        self.dropout = nn.Dropout(0.1)
	        
	        # relu activation function
	        self.relu =  nn.ReLU()
	        # dense layer 1
	        if args['mid_layer']:
	        	self.fc1 = nn.Linear(3,args['mid_layer_size'])

	        	self.fc2 = nn.Linear(args['mid_layer_size'],2)
	        else:
	        	self.fc1 = nn.Linear(3,2)

	        #softmax activation function
	        self.softmax = nn.LogSoftmax(dim=1)

	    #define the forward pass
	    def forward(self, input_ids,
	            attention_mask,
	            token_type_ids,labels):

	        #pass the inputs to the model  
	        outputs = self.model(input_ids,
	            attention_mask=attention_mask,
	            token_type_ids=token_type_ids,labels = labels)
	        
	        cls_hs = outputs.logits

	        if args['mid_layer']:
	        	x = self.fc1(cls_hs)

		        x = self.relu(x)

		        x = self.dropout(x)

		        # output layer
		        x = self.fc2(x)
		        
		        # apply softmax activation
		        x = self.softmax(x)
	        else:
		        x = self.dropout(cls_hs)

		        # output layer
		        x = self.fc1(x)
		        
		        # apply softmax activation
		        x = self.softmax(x)

	        return x


	# In[74]:


	model_full = BERT_Arch(model)


	# In[81]:


	from torch.utils.data import DataLoader
	from transformers import AdamW
	from torch.nn import DataParallel


	model_full.to(device)
	model_full.train()

	model_full = DataParallel(model_full)


	train_loader = DataLoader(train_dataset, 
		batch_size=args['batch_size'], 
		shuffle=True)

	optim = AdamW(model.parameters(), lr=args['learning_rate'])


	# In[83]:


	cross_entropy  = nn.NLLLoss() 
	for epoch in range(args['epochs']):
	    for i,batch in enumerate(Bar(train_loader)):
	        optim.zero_grad()
	        input_ids = batch['input_ids'].to(device, dtype = torch.long)
	        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
	        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
	        labels = batch['labels'].to(device, dtype = torch.long)
	        outputs = model_full(input_ids, 
	                        attention_mask=attention_mask, 
	                        token_type_ids = token_type_ids,
	                        labels = labels)
	        #loss = outputs.loss
	        loss = cross_entropy(outputs, labels)
	        loss.backward()
	        optim.step()
	model_full.eval()


	# In[ ]:
	if args['mid_layer']:
		checkpoint_output = args['checkpoint_output_path'] + '/checkpoint_yn_' + str(args['mid_layer_size']) + '.pt'
	else:
		checkpoint_output = args['checkpoint_output_path'] + '/checkpoint_yn_direct.pt'

	torch.save({
	            'epoch': args['epochs'],
	            'model_state_dict': model_full.state_dict(),
	            'optimizer_state_dict': optim.state_dict(),
	            'loss': loss,
	            },
	            checkpoint_output)










if __name__ == "__main__":
    main()