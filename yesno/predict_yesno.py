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
	parser.add_argument("--predictions_output_path", required= True)
	parser.add_argument("--questions_path", required= True)
	
	parser.add_argument('--mid_layer', dest='mid_layer', action='store_true')
	parser.add_argument('--no-mid_layer', dest='mid_layer', action='store_false')
	parser.set_defaults(mid_layer=True)

	parser.add_argument('--balance', dest='balance', action='store_true')
	parser.add_argument('--no-balance', dest='balance', action='store_false')
	parser.set_defaults(balance=True)

	parser.add_argument("--mid_layer_size", default = 256, type = int)
	args = vars(parser.parse_args())

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


	from transformers import BertForSequenceClassification

	model = BertForSequenceClassification.from_pretrained(args['model_name'], num_labels = 3)

	from torch import nn
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
	            token_type_ids,labels=None):

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


	model_full = BERT_Arch(model)

	checkpoint = torch.load(args['checkpoint_input_path'],map_location=device)

	checkpoint_model_dict = { k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}

	model_full.load_state_dict(checkpoint_model_dict)

	from transformers import BertTokenizer
	# Load the BERT tokenizer.
	tokenizer = BertTokenizer.from_pretrained(args['model_name'], 
	                                          do_lower_case=True)


	from torch.nn import Softmax

	def get_yn_predictions_csv(questions_path,output_path):    
    

	    with open(questions_path, 'rb') as f:
	        bio_yn_raw = json.load(f)['questions']

	    bio_yn_raw = [question for question in bio_yn_raw if question['type'] == 'yesno']

	    bio_yn_questions = [question['body'] for question in bio_yn_raw]
	    bio_yn_ids = [question['id'] for question in bio_yn_raw]
	    bio_snippets = {question['id'] : [snippet['text'] 
	                                      for snippet in question['snippets']] 
	                    for question in bio_yn_raw}

	    ids = []
	    snippets = []
	    for key, value in bio_snippets.items():
	        for snippet in value:
	            ids.append(key)
	            snippets.append(snippet)

	    snippets_df = pd.DataFrame({'id': ids,'snippet': snippets})
	    questions_df = pd.DataFrame({'id': bio_yn_ids, 
	                                 'question': bio_yn_questions})#,
	                                #'label': bio_yn_labels})
	    val_df = pd.merge(snippets_df,questions_df, how = 'left', on = 'id')
	    val_a = list(val_df.question)
	    val_b = list(val_df.snippet)

	    val_tokens = tokenizer(val_a,val_b, 
	                           add_special_tokens=True,
	                           max_length=500,
	                           truncation=True, padding=True,return_tensors='pt')

	    p = Softmax(dim = 1)
	    val_predictions = []
	    val_prob = []
	    
	    for i in range(len(val_a)):    
	        inputs = tokenizer(val_a[i], val_b[i], 
	                               add_special_tokens=True,
	                               max_length=500,
	                               truncation=True, padding=True,return_tensors='pt')
	        output = model_full(**inputs)
	        prob = float(torch.max(p(output)))
	        pred = torch.argmax(p(output))
	        val_prob.append(prob)
	        val_predictions.append(int(pred))

	    res = pd.DataFrame({'id': val_df.id,
	                        'pred': val_predictions,
	                        'prob': val_prob})
	    def get_p1(row):
	    
	        if row['pred'] == 1:
	            return row['prob']
	        return 1 - row['prob']

	    def get_p0(row):

	        if row['pred'] == 0:
	            return row['prob']
	        return 1 - row['prob']
	    
	    res['p1'] = res.apply(get_p1,axis = 1)
	    res['p0'] = res.apply(get_p0,axis = 1)
	    
	    def get_pred(row):
	        if row['p1'] > row['p0']:
	            return 1
	        return 0
	    def get_prob(row):
	        if row['pred'] == 1:
	            return row['p1']
	        return row['p0']
	    
	    pred = res.groupby(['id']).mean()[['p1','p0']].reset_index()
	    pred['pred'] = pred.apply(get_pred, axis = 1)
	    pred['prob'] = pred.apply(get_prob, axis = 1)
	    pred = pred[['id','pred','prob']]
	    pred.to_csv(output_path, index = False)


	get_yn_predictions_csv(args['questions_path'],args['predictions_output_path'])

if __name__ == "__main__":
    main()