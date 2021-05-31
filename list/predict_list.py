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
	parser.add_argument("--checkpoint_input_path", required= True)
	parser.add_argument("--predictions_output_path", required= True)
	parser.add_argument("--questions_path", required= True)
	parser.add_argument("--k_candidates", required= False, type = int, default = 5)
	parser.add_argument("--k_elected", required= False, type = int, default = 2)
	parser.add_argument("--voting", required= False, type = str, default = 'stv')

	parser.add_argument('--hopeful', dest='hopeful', action='store_true')
	parser.add_argument('--no-hopeful', dest='hopeful', action='store_false')
	parser.set_defaults(hopeful=True)


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


	def get_questions(path):
	    with open(path, 'rb') as f:
	        bio_factoid_raw = json.load(f)['questions']

	    bio_list_raw = [question for question in bio_factoid_raw if question['type'] == 'list']

	    bio_list_questions = [question['body'] for question in bio_list_raw]
	    bio_list_ids = [question['id'] for question in bio_list_raw]
	    bio_list_answers = [question['exact_answer']for question in bio_list_raw]
	    bio_snippets = {question['id'] : [snippet['text'] 
	                                      for snippet in question['snippets']] 
	                    for question in bio_list_raw}
	    print(f'Number of questions: {len(bio_list_questions)}')
	    
	    ids = []
	    snippets = []
	    for key, value in bio_snippets.items():
	        for snippet in value:
	            ids.append(key)
	            snippets.append(snippet)

	    snippets_df = pd.DataFrame({'id': ids,'snippet': snippets})
	    questions_df = pd.DataFrame({'id': bio_list_ids, 
	                                 'question': bio_list_questions,
	                                'label': bio_list_answers})
	    val_df = pd.merge(snippets_df,questions_df, how = 'left', on = 'id')
	    
	    ids = np.array(val_df.id)
	    questions = np.array(val_df.question)
	    contexts = np.array(val_df.snippet)
	    return ids,questions,contexts


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
	    while (len(possibles) < n_ans) & (i< total_scores.shape[0]):
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
	        answers_dict['id']+= [ids[i]+f'_{i}']*n
	        answers_dict['answers']+=answers_list[i]
	        answers_dict['scores']+=scores_list[i]
	    def clean_text(x):
	        x = x.replace('( ','(')
	        x = x.replace(' )',')')
	        x = x.replace(' - ','-')
	        x = x.replace('.','')
	        x = x.replace('[ ','[')
	        x = x.replace(' ]',']')
	        return x
	    pred_df = pd.DataFrame.from_dict(answers_dict)
	    pred_df.scores = pred_df.scores.map(lambda x: float(x))
	    pred_df.answers = pred_df.answers.map(lambda x: clean_text(str(x)))
	    pred_df = pred_df.sort_values(by = ['id','scores'], ascending = False).drop_duplicates(['id','answers'])
	    return pred_df

	import pyrankvote
	from pyrankvote import Candidate, Ballot

	def get_election_res(pred_df,id_orig, n_ans):
	    df = pred_df[pred_df.id.str.startswith(id_orig)]
	    candidates =[Candidate(el) for el in df.answers.unique()]
	    ballots = [Ballot(ranked_candidates=[Candidate(vote) 
	                                         for vote in df[df.id == id_].answers]) 
	               for id_ in df.id.unique()]

	    if len(candidates) <= n_ans:
	        return list(df.answers)
	    if args['voting']=='pbv':
	    	election_result = pyrankvote.preferential_block_voting(candidates, ballots,min(n_ans,len(candidates)))
	    else:
	    	election_result = pyrankvote.single_transferable_vote(candidates, ballots,min(n_ans,len(candidates)))

	    return election_result

	def get_pred_election(elect_res, hopeful = False):
	    if type(elect_res) == list:
	        return elect_res
	    
	    elected = [candidate.candidate.name 
	               for candidate in elect_res.rounds[-1].candidate_results 
	               if candidate.status == 'Elected']
	    if len(elect_res.rounds) > 1:
	        hope = [candidate.candidate.name 
	               for candidate in elect_res.rounds[-2].candidate_results 
	               if candidate.status in ['Hopeful','Elected']]
	    if hopeful & (len(elect_res.rounds) > 1):
	        return hope
	    return elected

	def get_final_preds(path,n_ans,n_elected,output_path,hopeful = False):
	    ids,questions,contexts = get_questions(path)
	    pred_df = get_batch_answers(ids,questions,contexts,n_ans)
	    pred_df = pred_df[pred_df.scores>0.2]
	    def remove_first_space(x):
	        if x[0] == ' ':
	            return x[1:]
	        return x
	    pred_rank_dict = {i:{} for i in pred_df.id.map(lambda x: x.split('_',1)[0]).unique()}
	    for q_id in pred_rank_dict.keys():
	        preds = get_pred_election(get_election_res(pred_df,q_id,n_elected),hopeful)
	        preds_clean = list(map(lambda x: x.replace('the ','').replace(' vs ',',').replace(' and ',',').replace(' or ',',').replace(' / ',',').replace(' plus ',',').split(","), preds))
	        list_answers = list(map(remove_first_space,[l[i] for l in preds_clean for i in range(len(l)) if l[i] != '']))
	        set_answers = list(set(list_answers))

	        pred_rank_dict[q_id]['answers'] = set_answers
	    list_ids_rank = [len(value['answers'])*[key] for key,value in pred_rank_dict.items()]
	    list_ids_rank = [item for sub in list_ids_rank for item in sub]
	    answers_rank = []
	    for key,value in pred_rank_dict.items():
	        answers_rank+=value['answers']
	    pred_df_final_rank = pd.DataFrame({'id': list_ids_rank,
	                                      'pred': answers_rank })
	    #return pred_df_final_rank
	    pred_df_final_rank.to_csv(output_path, index = False)

	get_final_preds(args['questions_path'],
		args['k_candidates'],
		args['k_elected'],
		args['predictions_output_path'],
		hopeful=args['hopeful'])

if __name__ == "__main__":
    main()



