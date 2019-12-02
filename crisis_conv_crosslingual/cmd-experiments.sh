#!/usr/bin/env bash


# experiments for ? : is crisis related?

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated4.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed.csv

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed.csv \
    --output-file=../results/pred_lr.csv \
    --labels=crisis_related \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed.csv \
    --output-file=../results/pred_cnn.csv \
    --labels=crisis_related \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed.csv \
    --output-file=../results/pred_lstm.csv \
    --labels=crisis_related \
    --model=lstm

# experiments for ? : is crisis related? only roots

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated_roots.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_roots.csv

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_roots.csv \
    --output-file=../results/pred_lr_roots.csv \
    --labels=crisis_related \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_roots.csv \
    --output-file=../results/pred_cnn_roots.csv \
    --labels=crisis_related \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_roots.csv \
    --output-file=../results/pred_lstm_roots.csv \
    --labels=crisis_related \
    --model=lstm


# experiments for ? : is crisis related? only replies

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated_replies.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_replies.csv

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_replies.csv \
    --output-file=../results/pred_lr_replies.csv \
    --labels=crisis_related \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_replies.csv \
    --output-file=../results/pred_cnn_replies.csv \
    --labels=crisis_related \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_replies.csv \
    --output-file=../results/pred_lstm_replies.csv \
    --labels=crisis_related \
    --model=lstm


# experiments for multilabels


python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated3.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lr_ml6.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_cnn_ml6.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lstm_ml6.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other \
    --model=lstm

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lr_ml8.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other,thanks,request_info \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_cnn_ml8.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other,thanks,request_info \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lstm_ml8.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other,thanks,request_info \
    --model=lstm

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lr_ml10.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other,thanks,request_info,suggest_action,complain \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_cnn_ml10.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other,thanks,request_info,suggest_action,complain \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lstm_ml10.csv \
    --labels=informative,expressive_positive,sarcasm,people_deaths,expressive_negative,response_other,thanks,request_info,suggest_action,complain \
    --model=lstm


python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lr_mln.csv \
    --labels=people_deaths,people_wounded,people_missing,people_other,infra_buildings,infra_roads,infra_houses,infra_business,infra_other,request_info,request_goods,request_services,request_other,offer_info,offer_goods,offer_services,offer_other,informative,update,expressive_positive,expressive_negative,complain,suggest_action,promise,sarcasm,yes_no_question,wh_question,open_question,yes_answer,no_answer,response_ack,response_other,opening_greeting,closing_greeting,thanks,apology,other_subcat \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_cnn_mln.csv \
    --labels=people_deaths,people_wounded,people_missing,people_other,infra_buildings,infra_roads,infra_houses,infra_business,infra_other,request_info,request_goods,request_services,request_other,offer_info,offer_goods,offer_services,offer_other,informative,update,expressive_positive,expressive_negative,complain,suggest_action,promise,sarcasm,yes_no_question,wh_question,open_question,yes_answer,no_answer,response_ack,response_other,opening_greeting,closing_greeting,thanks,apology,other_subcat \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_ml.csv \
    --output-file=../results/pred_lstm_mln.csv \
    --labels=people_deaths,people_wounded,people_missing,people_other,infra_buildings,infra_roads,infra_houses,infra_business,infra_other,request_info,request_goods,request_services,request_other,offer_info,offer_goods,offer_services,offer_other,informative,update,expressive_positive,expressive_negative,complain,suggest_action,promise,sarcasm,yes_no_question,wh_question,open_question,yes_answer,no_answer,response_ack,response_other,opening_greeting,closing_greeting,thanks,apology,other_subcat \
    --model=lstm



# OUTCOMES

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated_outcomes.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes.csv

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated_outcomes6c.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes6c.csv

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated_outcomes8c.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes8c.csv

python3 -m preprocessing.data_tokenizer \
    --input-file=../data/ecuador_earthquake_2016/es/conversations_annotated_outcomes10c.csv \
    --output-file=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes10c.csv


python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes.csv \
    --output-file=../results/pred_lr_outcomes.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes.csv \
    --output-file=../results/pred_cnn_outcomes.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes.csv \
    --output-file=../results/pred_lstm_outcomes.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lstm

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes6c.csv \
    --output-file=../results/pred_lr_outcomes6c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes6c.csv \
    --output-file=../results/pred_cnn_outcomes6c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes6c.csv \
    --output-file=../results/pred_lstm_outcomes6c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lstm

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes8c.csv \
    --output-file=../results/pred_lr_outcomes8c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes8c.csv \
    --output-file=../results/pred_cnn_outcomes8c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes8c.csv \
    --output-file=../results/pred_lstm_outcomes8c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lstm

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes10c.csv \
    --output-file=../results/pred_lr_outcomes10c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lr

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes10c.csv \
    --output-file=../results/pred_cnn_outcomes10c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=cnn

python3 -m task \
    --data-labeled=../data/ecuador_earthquake_2016/es/conversations_preprocessed_outcomes10c.csv \
    --output-file=../results/pred_lstm_outcomes10c.csv \
    --labels=outcome_prevention,outcome_awareness,outcome_relief \
    --model=lstm



# --------------

DATA_DIR=../data/ecuador_earthquake_2016/
RESULTS_DIR=../results/


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lr_es.csv \
    --labels=crisis_related \
    --model=lr \
    --kfolds=10


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_lr_en.csv \
    --labels=crisis_related \
    --model=lr \
    --kfolds=10



python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_cnn_es.csv \
    --labels=crisis_related \
    --model=cnn \
    --kfolds=10

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_cnn_en.csv \
    --labels=crisis_related \
    --model=cnn \
    --kfolds=10

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_es.csv \
    --labels=crisis_related \
    --model=lstm \
    --kfolds=10

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_en.csv \
    --labels=crisis_related \
    --model=lstm \
    --kfolds=10

#-----

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_lr_esen.csv \
    --labels=crisis_related \
    --model=lr \
    --kfolds=10 \
    --predict


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lr_enes.csv \
    --labels=crisis_related \
    --model=lr \
    --kfolds=10 \
    --predict


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_cnn_esen.csv \
    --labels=crisis_related \
    --model=cnn \
    --kfolds=10 \
    --predict


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_cnn_enes.csv \
    --labels=crisis_related \
    --model=cnn \
    --kfolds=10 \
    --predict


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_esen.csv \
    --labels=crisis_related \
    --model=lstm \
    --kfolds=10 \
    --predict


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_enes.csv \
    --labels=crisis_related \
    --model=lstm \
    --kfolds=10 \
    --predict


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_se_es.csv \
    --labels=crisis_related \
    --model=bilstm \
    --kfolds=10

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_se_en.csv \
    --labels=crisis_related \
    --model=bilstm \
    --kfolds=10


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_se_esen.csv \
    --labels=crisis_related \
    --model=bilstm \
    --kfolds=10 \
    --predict

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_en.csv \
    --data-test=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lstm_se_enes.csv \
    --labels=crisis_related \
    --model=bilstm \
    --kfolds=10 \
    --predict


#------

python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --output-file=${RESULTS_DIR}/predictions_lr_full_es.csv \
    --labels=choose_one_category \
    --model=lr \
    --kfolds=10


python -m task \
    --data-labeled=${DATA_DIR}/2016_ecuador_eq_es.csv \
    --data-test=${DATA_DIR}/en/conversations.csv \
    --output-file=${RESULTS_DIR}/conversations_predictions_en.csv \
    --labels=choose_one_category \
    --model=lr \
    --kfolds=2 \
    --predict