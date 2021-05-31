import numpy as np 
import pandas as pd
import json
import argparse

def main():

	print('Start')
	parser = argparse.ArgumentParser()

	# Add the arguments to the parser
	parser.add_argument("--questions_path", required= True)
	parser.add_argument("--list_predictions_path", required= False,default = '')
	parser.add_argument("--factoid_predictions_path", required= False,default = '')
	parser.add_argument("--yesno_predictions_path", required= False,default = '')
	parser.add_argument("--output_path", required= True)


	args = vars(parser.parse_args())


	def get_bioasq_pred(gold_path,list_path,yn_path,factoid_path,output_path):
    
	    with open(gold_path, 'rb') as f:
	        bio = json.load(f)

	    yn_pred = pd.read_csv(args['yesno_predictions_path'],index_col='id')
	    factoid_pred = pd.read_csv(args['factoid_predictions_path'],index_col='id')
	    list_pred = pd.read_csv(args['list_predictions_path'],index_col='id')
	    
	    pred = bio

	    if args['yesno_predictions_path'] != '':
		    for q in pred['questions']:
		        if q['type'] == 'yesno':
		            if yn_pred.loc[q['id']].pred == 1:
		                q['exact_answer'] = 'yes'
		            else:
		                q['exact_answer'] = 'no'
		            q['ideal_answer'] = 'dummy'
	      
	    if args['factoid_predictions_path'] != '':
		    for q in pred['questions']:
		        if q['type'] == 'factoid':
		            q['exact_answer'] = [[i] for i in factoid_pred.loc[q['id']].answers]
		            q['ideal_answer'] = 'dummy'
	          
	    if args['list_predictions_path'] != '':
		    for q in pred['questions']:
		        if q['type'] == 'list':
		            q['exact_answer'] = [[i] for i in list_pred.loc[q['id']].pred]
		            q['ideal_answer'] = 'dummy'      

	    with open(output_path, 'w') as fp:
		    json.dump(pred, fp)


	get_bioasq_pred(args['questions_path'],
		args['list_predictions_path'],
		args['yesno_predictions_path'],
		args['factoid_predictions_path'],
		args['output_path'])




if __name__ == "__main__":
    main()
