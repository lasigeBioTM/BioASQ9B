import argparse
import json
import numpy as np
import pandas as pd
import pickle
import torch
import random
import jsonlines
from pathlib import Path


def main():

	print('Start')
	parser = argparse.ArgumentParser()

	# Add the arguments to the parser
	parser.add_argument("--model_name", required= True)
	parser.add_argument("--checkpoint_input_path", required= False)
	parser.add_argument("--predictions_output_path", required= True)
	parser.add_argument("--questions_path", required= True)
	parser.add_argument("--k_candidates", required= True, type = int)

	args = vars(parser.parse_args())
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


	from transformers import BertTokenizerFast, BertModel


	tokenizer_fast = BertTokenizerFast.from_pretrained(args['model_name'], 
	                                          do_lower_case=True,
	                                          padding = True,
	                                          truncation=True,
	                                          add_special_tokens = True,
	                                          model_max_length = 1000000000)

	from transformers import BertForQuestionAnswering

	model = BertForQuestionAnswering.from_pretrained(args['model_name'])
	checkpoint = torch.load(args['checkpoint_input_path'],map_location=device)
	model.load_state_dict({key.replace('module.',''): value for key,value in checkpoint['model_state_dict'].items()})


	def get_questions_df(path):

	    with open(path, 'rb') as f:
	        bio_factoid_raw = json.load(f)['questions']

	    bio_factoid_raw = [question for question in bio_factoid_raw if question['type'] == 'factoid']

	    bio_factoid_questions = [question['body'] for question in bio_factoid_raw]
	    bio_factoid_ids = [question['id'] for question in bio_factoid_raw]
	    
	    #bio_factoid_answers = [question['exact_answer'][0][0] for question in bio_factoid_raw]
	    bio_snippets = {question['id'] : [snippet['text'] 
	                                      for snippet in question['snippets']] 
	                    for question in bio_factoid_raw}
	    print(f'Number of questions: {len(bio_factoid_questions)}')

	    ids = []
	    snippets = []
	    for key, value in bio_snippets.items():
	        for snippet in value:
	            ids.append(key)
	            snippets.append(snippet)

	    snippets_df = pd.DataFrame({'id': ids,'snippet': snippets})
	    questions_df = pd.DataFrame({'id': bio_factoid_ids, 
	                                 'question': bio_factoid_questions})#,
	                                #'label': bio_factoid_answers})
	    val_df = pd.merge(snippets_df,questions_df, how = 'left', on = 'id')

	    return val_df

	from torch.nn import Softmax



	def get_possible_answers(question,context,n_ans):
	    p = Softmax(dim = 0)
	    inputs = tokenizer_fast(question, 
	                            context,
	                           truncation=True, 
	                           padding=True,
	                           max_length=500, 
	                           return_tensors="pt")
	    input_ids = inputs["input_ids"].tolist()[0]
	    context_ids = np.array(input_ids)[np.array(inputs['token_type_ids'][0], dtype = bool)]
	    text_tokens = tokenizer_fast.convert_ids_to_tokens(input_ids)
	    outputs = model(**inputs)
	    answer_start_scores = outputs.start_logits[0][torch.tensor(inputs['token_type_ids'][0], dtype = bool)][:-1]
	    answer_end_scores = outputs.end_logits[0][torch.tensor(inputs['token_type_ids'][0], dtype = bool)][:-1]
	    total_scores = np.array([
	            [float(answer_start_scores[i] + answer_end_scores[j]) 
	                 for j in range(len(answer_start_scores))]
	             for i in range(len(answer_start_scores))])
	    possibles = []
	    possible_scores = []
	    
	    i = 0
	    while (len(possibles) < n_ans) & (i < total_scores.shape[0]):

	        row,col = np.unravel_index(np.argsort(total_scores.ravel()),total_scores.shape)
	        row,col = row[::-1],col[::-1]

	        start = row[i]
	        end = col[i]


	        if (end >= start) & (tokenizer_fast.convert_ids_to_tokens(context_ids[start:(start+1)])[0][0:2] != "##")&\
	            (tokenizer_fast.convert_ids_to_tokens(context_ids[(end+1):(end+2)])[0][0:2] != "##"):
	            score = torch.tensor(total_scores)[start,end]
	            answer = tokenizer_fast.convert_tokens_to_string(
	                        tokenizer_fast.convert_ids_to_tokens(context_ids[start:(end+1)]))
	            if (len(answer) <= 100) & (1 - ("(" in answer) & ~(")" in answer)) & (~(answer in possibles)):
	                possibles.append(answer)
	                possible_scores.append(score)
	        i+=1
	    possible_scores_prob = p(torch.tensor(possible_scores))
	    return possibles, possible_scores_prob

	def get_batch_answers(ids,questions,contexts,n_ans):

	    answers_list = []
	    scores_list = []
	    for i in range(len(questions)):
	        answers,scores = get_possible_answers(questions[i],contexts[i],n_ans)
	        answers_list.append(answers)
	        scores_list.append(scores)
	        
	    answers_dict = {'id': [],
	                   'answers': [],
	                   'scores' : []}
	    for i in range(questions.shape[0]):
	        n = len(answers_list[i])
	        answers_dict['id']+= [ids[i]]*n
	        answers_dict['answers']+=answers_list[i]
	        answers_dict['scores']+=scores_list[i]
	    
	    pred_df = pd.DataFrame.from_dict(answers_dict)
	    pred_df.scores = pred_df.scores.map(lambda x: float(x))
	    
	    return pred_df

	def get_top_5(pred_df):
	    def clean_text(x):
	        x = x.replace('( ','(')
	        x = x.replace(' )',')')
	        x = x.replace(' - ','-')
	        x = x.replace('.','')
	        x = x.replace('[ ','[')
	        x = x.replace(' ]',']')
	        x = x.replace(' / ','/')
	        x = x.replace(' * ','*')
	        x = x.replace(' : ',':')
	        x = x.replace(' + ','+')
	        x = x.replace(' %','%')
	        return x
	    pred_df.answers = pred_df.answers.map(lambda x: clean_text(str(x)))
	    top_5_preds = pred_df \
	                    .sort_values(['id','scores'], ascending = False) \
	                    .drop_duplicates(['id','answers']) \
	                    .groupby('id') \
	                    .head(5)[['id','answers','scores']]
	    return top_5_preds


	def get_top_batch(path,n_ans):
	    val_df = get_questions_df(path)
	    ids = np.array(val_df.id)
	    questions = np.array(val_df.question)
	    contexts = np.array(val_df.snippet)
	    pred_df = get_batch_answers(ids,questions,contexts,n_ans)
	    top_5_pred = get_top_5(pred_df)
	    return top_5_pred

	pred_df = get_top_batch(args['questions_path'],args['k_candidates'])
	pred_df.to_csv(args['predictions_output_path'])

if __name__ == "__main__":
    main()

