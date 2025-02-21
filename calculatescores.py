import formatresultshelper
import numpy as np

from sklearn.metrics import  roc_auc_score

#Defined as utilized documents / retrieved documents for the query
def compute_context_relevance(relevant_sentences, support_keys):
    total_relevance_score = 0
    total_relevant_sentences = len(relevant_sentences)

    for sentence in relevant_sentences:
      if sentence in support_keys:
        total_relevance_score += 1

    # To avoid division by zero in case there are no relevant sentences
    if total_relevant_sentences == 0:
        return 0

    return total_relevance_score / total_relevant_sentences

def compute_context_utilization(relevant_sentences, utilization_levels):
    total_utilization_score = 0
    total_relevant_sentences = len(relevant_sentences)
    for sentence in relevant_sentences:
      if sentence in utilization_levels:
        total_utilization_score += 1
    # To avoid division by zero in case there are no relevant sentences
    if total_relevant_sentences == 0:
        return 0
    return total_utilization_score / total_relevant_sentences


def CalculateScores(relevant_sentence_keys,all_utilized_sentence_keys,support_keys,support_level,extarcted_row_for_given_id):
   #compute Context Relevance
   contextrel = compute_context_relevance(relevant_sentence_keys, support_keys)
   print(f"Compute Context Relevance = {contextrel}")

   contextutil = compute_context_utilization(relevant_sentence_keys, all_utilized_sentence_keys)
   print(f"Compute Context Utilization = {contextutil}")

   compnum = np.intersect1d(support_keys, all_utilized_sentence_keys)
   completenes = compnum.size / len(support_keys)
   print(f"Completeness = {completenes}")

   #Adherence : whether all parts of response are grounded by context
   for val in support_level:
     prevval = 1;
     if val == False:
       adherence = 0 * prevval
       break
     else:
       adherence = 1 * prevval
     prevval = adherence

   print(f"Adherence = {adherence}")

   context_relevance_score = extarcted_row_for_given_id['relevance_score'].values[0]
   context_utilization_score = extarcted_row_for_given_id['utilization_score'].values[0]
   adherence_score = float(extarcted_row_for_given_id['adherence_score'].values[0])
   docadherencearr = np.array([adherence_score, 0, 0])
   adherencearr = np.array([adherence, 0, 0])
   rmsecontextrel = mse(context_relevance_score, contextrel)
   rmsecontextutil = mse(context_utilization_score, contextutil)
   aucscore = roc_auc_score(docadherencearr, adherencearr)

   return rmsecontextrel, rmsecontextutil, aucscore



def mse(actual, predicted):
    return (actual - predicted)**2   