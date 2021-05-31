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
	parser.add_argument("--checkpoint_output_path", required= True)
	parser.add_argument("--bioasq_path", required= True)
	parser.add_argument("--seed", default = 1995)
	parser.add_argument("--learning_rate", default = 5e-5, type = float)
	parser.add_argument("--batch_size", default = 16, type = int)
	parser.add_argument("--epochs", default = 3, type=int)





	args = vars(parser.parse_args())




	random.seed(args['seed'])
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



	with open(args['bioasq_path'], 'rb') as f:
		bio_factoid_raw = json.load(f)['questions']

	bio_factoid_raw = [question for question in bio_factoid_raw if question['type'] == 'factoid']

	bio_factoid_questions = [question['body'] for question in bio_factoid_raw]
	bio_factoid_ids = [question['id'] for question in bio_factoid_raw]
	bio_factoid_answers = [question['exact_answer'][0] for question in bio_factoid_raw]
	bio_snippets = {question['id'] : [snippet['text'] 
	                                  for snippet in question['snippets']] 
	                for question in bio_factoid_raw}
	ids = []
	snippets = []
	for key, value in bio_snippets.items():
	    for snippet in value:
	        ids.append(key)
	        snippets.append(snippet)

	snippets_df = pd.DataFrame({'id': ids,'snippet': snippets})
	questions_df = pd.DataFrame({'id': bio_factoid_ids, 
	                             'question': bio_factoid_questions,
	                            'label': bio_factoid_answers})
	val_df = pd.merge(snippets_df,questions_df, how = 'left', on = 'id')

	val_df = val_df.sample(16)


	def get_start_answer(row):
	    label = row['label'].lower()
	    context = row['snippet'].lower()
	    if label in context:
	        return context.index(label)
	    return None

	val_df['answer_start'] = val_df.apply(get_start_answer, axis = 1)
	clean_df = val_df[~val_df.answer_start.isnull()]

	bio_factoid_questions = list(clean_df.question)
	bio_factoid_contexts = list(clean_df.snippet)

	bio_factoid_answers = [{'text': row['label'],
	                       'answer_start': int(row['answer_start'])} 
	                      for index, row in clean_df.iterrows()]

	from transformers import BertTokenizer, BertTokenizerFast


	tokenizer_fast = BertTokenizerFast.from_pretrained(args['model_name'], 
	                                          do_lower_case=True,
	                                          padding = True,
	                                          truncation=True,
	                                          add_special_tokens = True,
	                                          model_max_length = 1000000000)

	from squad_processing import add_end_idx, add_token_positions

	add_end_idx(bio_factoid_answers,bio_factoid_contexts)

	bio_factoid_encodings = tokenizer_fast(bio_factoid_contexts,
		bio_factoid_questions,
    	add_special_tokens=True,
	    truncation=True,
	    padding=True,
	    max_length=500)

	# Processing of token positions
	add_token_positions(bio_factoid_encodings, bio_factoid_answers,tokenizer_fast)


	from torch.utils.data import Dataset

	class SquadDataset(Dataset):
	    def __init__(self, encodings):
	        self.encodings = encodings

	    def __getitem__(self, idx):
	        #print(self.encodings['start_positions'][idx])
	         #{key: torch.tensor(val[idx], dtype = torch.long) for key, val in self.encodings.items()}
	        return {'input_ids':torch.tensor(self.encodings['input_ids'][idx],dtype = torch.long),
	         'attention_mask':torch.tensor(self.encodings['attention_mask'][idx],dtype = torch.long),
	         'start_positions':torch.tensor(self.encodings['start_positions'][idx],dtype = torch.long),
	         'end_positions':torch.tensor(self.encodings['end_positions'][idx],dtype = torch.long)}

	    def __len__(self):
	        return len(self.encodings.input_ids)

	train_bio_factoid = SquadDataset(bio_factoid_encodings)

	from transformers import BertPreTrainedModel, BertModel
	from torch import nn
	from torch.nn import CrossEntropyLoss


	from torch.nn import DataParallel
	from torch.utils.data import DataLoader
	from transformers import AdamW


	from transformers.modeling_outputs import QuestionAnsweringModelOutput


	class BertForQuestionAnswering(BertPreTrainedModel):

	    _keys_to_ignore_on_load_unexpected = [r"pooler"]

	    def __init__(self, config):
	        super().__init__(config)
	        self.num_labels = config.num_labels

	        self.bert = BertModel(config, add_pooling_layer=False)
	        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

	        self.init_weights()

	    def forward(
	        self,
	        input_ids=None,
	        attention_mask=None,
	        token_type_ids=None,
	        position_ids=None,
	        head_mask=None,
	        inputs_embeds=None,
	        start_positions=None,
	        end_positions=None,
	        output_attentions=None,
	        output_hidden_states=None,
	        return_dict=None,
	    ):
	        r"""
	        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
	            Labels for position (index) of the start of the labelled span for computing the token classification loss.
	            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
	            sequence are not taken into account for computing the loss.
	        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
	            Labels for position (index) of the end of the labelled span for computing the token classification loss.
	            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
	            sequence are not taken into account for computing the loss.
	        """
	        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

	        outputs = self.bert(
	            input_ids,
	            attention_mask=attention_mask,
	            token_type_ids=token_type_ids,
	            position_ids=position_ids,
	            head_mask=head_mask,
	            inputs_embeds=inputs_embeds,
	            output_attentions=output_attentions,
	            output_hidden_states=output_hidden_states,
	            return_dict=return_dict,
	        )

	        sequence_output = outputs[0]

	        logits = self.qa_outputs(sequence_output)
	        start_logits, end_logits = logits.split(1, dim=-1)
	        start_logits = start_logits.squeeze(-1)
	        end_logits = end_logits.squeeze(-1)

	        total_loss = None
	        if start_positions is not None and end_positions is not None:
	            # If we are on multi-GPU, split add a dimension
	            if len(start_positions.size()) > 1:
	                start_positions = start_positions.squeeze(-1)
	            if len(end_positions.size()) > 1:
	                end_positions = end_positions.squeeze(-1)
	            # sometimes the start/end positions are outside our model inputs, we ignore these terms
	            ignored_index = start_logits.size(1)
	            start_positions.clamp_(0, ignored_index)
	            end_positions.clamp_(0, ignored_index)

	            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
	            start_loss = loss_fct(start_logits, start_positions)
	            end_loss = loss_fct(end_logits, end_positions)
	            total_loss = (start_loss + end_loss) / 2

	        if not return_dict:
	            output = (start_logits, end_logits) + outputs[2:]
	            return ((total_loss,) + output) if total_loss is not None else output

	        return QuestionAnsweringModelOutput(
	            loss=total_loss,
	            start_logits=start_logits,
	            end_logits=end_logits,
	            hidden_states=outputs.hidden_states,
	            attentions=outputs.attentions,
	        )


	train_loader_factoid = DataLoader(train_bio_factoid, 
		batch_size=args['batch_size'], 
		shuffle=True)

	squad_model = BertForQuestionAnswering.from_pretrained(args['model_name'])

	checkpoint = torch.load(args['checkpoint_input_path'], map_location = device)

	squad_model.load_state_dict({key.replace('module.',''): value for key,value in checkpoint.items()})

	squad_model = DataParallel(squad_model)

	squad_model.to(device)
	squad_model.train()
	optim = AdamW(squad_model.parameters(), lr=args['learning_rate'])

	from barbar import Bar



	for epoch in range(args['epochs']):
	    for i,batch in enumerate(Bar(train_loader_factoid)):
	        optim.zero_grad()
	        input_ids = batch['input_ids'].to(device, dtype = torch.long)
	        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
	        start_positions = batch['start_positions'].to(device, dtype = torch.long)
	        end_positions = batch['end_positions'].to(device, dtype = torch.long)
	        outputs = squad_model(input_ids, 
	                        attention_mask=attention_mask, 
	                        start_positions=start_positions, 
	                        end_positions=end_positions)
	        loss = outputs[0].to(device)
	        loss.sum().backward()
	        optim.step()
	squad_model.eval()

	print('Saving...')

	torch.save({
	            'epoch': args['epochs'],
	            'model_state_dict': squad_model.state_dict(),
	            'optimizer_state_dict': optim.state_dict(),
	            'loss': loss,
	            },#'checkpoint_mnli_squad_factoid_3epochs_seed_2.pt')
	            args['checkpoint_output_path'] + '/checkpoint_factoid.pt')



if __name__ == "__main__":
    main()






    