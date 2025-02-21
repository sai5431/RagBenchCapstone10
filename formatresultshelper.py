import generationhelper
import json

def evaluate_response_with_prompt(templete, query, documents, answer, eval_model="llama-3.3-70b-specdec"):

    formatted_documents = ""
    for doc_idx, doc_text in enumerate(documents["document"]):
      if isinstance(doc_text, list):
          doc_text = " ".join(doc_text)  # Convert list to a single string
      sentences = doc_text.split('. ')
      formatted_documents += "\n".join([f"{doc_idx}{chr(97 + i)}. {sent}" for i, sent in enumerate(sentences)]) + "\n"

    # Format response with unique keys (a, b, c)
    formatted_answer = "\n".join([f"{chr(97 + i)}. {sent}" for i, sent in enumerate(answer.split('. '))])

    prompt = templete.format(documents=formatted_documents, question=query, answer=formatted_answer)

    # Call the LLM API (Llama 3.3-70B)
    completion = generationhelper.groq_client.chat.completions.create(
        model=eval_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
        top_p=1
    )

    print("\nGenerated Response:\n", completion)

    return completion

def FormatAndScores(query, documents, answer, eval_model):
   templete= get_templet_to_calculatescores()

   completion_results = evaluate_response_with_prompt(templete, query,documents, answer, eval_model)

   print(completion_results)
   completion_results_response = completion_results.choices[0].message.content
   completion_results_response = completion_results_response.strip().strip('```')
   print(completion_results_response)

   # Check if response_content is empty
   if not completion_results_response.strip():
      raise ValueError("Empty response content")
   
   # Decode if it's a byte string
   if isinstance(completion_results_response, bytes):
      completion_results_response = completion_results_response.decode('utf-8')

   # Try to parse JSON
   try:
        data_json = json.loads(completion_results_response)
        print("JSON parsed successfully:")
        print(data_json)
   except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Response content: {completion_results_response}")

   relavance_explanation = data_json['relevance_explanation']
   relevant_sentence_keys = data_json['all_relevant_sentence_keys']
   overall_supported_explanation = data_json['overall_supported_explanation']
   overall_supported = data_json['overall_supported']
   sentence_support_information = data_json['sentence_support_information']
   all_utilized_sentence_keys = data_json['all_utilized_sentence_keys']

   support_keys = []
   support_level = []
   for sentence_support in sentence_support_information:
     support_keys += sentence_support['supporting_sentence_keys']
     support_level.append(sentence_support['fully_supported'])

   print(relavance_explanation)
   print(relevant_sentence_keys)
   print(overall_supported_explanation)
   print(overall_supported)
   print(sentence_support_information)
   print(all_utilized_sentence_keys)

   return completion_results_response,relevant_sentence_keys,all_utilized_sentence_keys,support_keys,support_level



def get_templet_to_calculatescores():
  return """
      You asked someone to answer a question based on one or more documents.
      Your task is to review their response and assess whether or not each sentence
      in that response is supported by text in the documents. And if so, which
      sentences in the documents provide that support. You will also tell me which
      of the documents contain useful information for answering the question, and
      which of the documents the answer was sourced from.
      Here are the documents, each of which is split into sentences. Alongside each
      sentence is an associated key, such as '0a.' or '0b.' that you can use to refer
      to it:
      ‘’’
      {documents}
      ‘’’
      The question was:
      ‘’’
      {question}
      ‘’’
      Here is their response, split into sentences. Alongside each sentence is
      an associated key, such as 'a.' or 'b.' that you can use to refer to it. Note
      that these keys are unique to the response, and are not related to the keys
      in the documents:
      ‘’’
      {answer}
      ‘’’
      You must respond with a JSON object matching this schema:
      ‘’’
      {{
      "relevance_explanation": string,
      "all_relevant_sentence_keys": [string],
      "overall_supported_explanation": string,
      "overall_supported": boolean,
      "sentence_support_information": [
      {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
      }},
      ],
      "all_utilized_sentence_keys": [string]
      }}
      ‘’’
      The relevance_explanation field is a string explaining which documents
      contain useful information for answering the question. Provide a step-by-step
      breakdown of information provided in the documents and how it is useful for
      answering the question.
      The all_relevant_sentence_keys field is a list of all document sentences keys
      (e.g. '0a') that are relevant to the question. Include every sentence that is
      useful and relevant to the question, even if it was not used in the response,
      or if only parts of the sentence are useful. Ignore the provided response when
      making this judgment and base your judgment solely on the provided documents
      and question. Omit sentences that, if removed from the document, would not
      impact someone's ability to answer the question.
      The overall_supported_explanation field is a string explaining why the response
      *as a whole* is or is not supported by the documents. In this field, provide a
      step-by-step breakdown of the claims made in the response and the support (or
      lack thereof) for those claims in the documents. Begin by assessing each claim
      separately, one by one; don't make any remarks about the response as a whole
      until you have assessed all the claims in isolation.
      The overall_supported field is a boolean indicating whether the response as a
      whole is supported by the documents. This value should reflect the conclusion
      you drew at the end of your step-by-step breakdown in overall_supported_explanation.
      In the sentence_support_information field, provide information about the support
      *for each sentence* in the response.
      The sentence_support_information field is a list of objects, one for each sentence
      in the response. Each object MUST have the following fields:
      - response_sentence_key: a string identifying the sentence in the response.
      This key is the same as the one used in the response above.
      - explanation: a string explaining why the sentence is or is not supported by the
      documents.
      - supporting_sentence_keys: keys (e.g. '0a') of sentences from the documents that
      support the response sentence. If the sentence is not supported, this list MUST
      be empty. If the sentence is supported, this list MUST contain one or more keys.
      In special cases where the sentence is supported, but not by any specific sentence,
      you can use the string "supported_without_sentence" to indicate that the sentence
      is generally supported by the documents. Consider cases where the sentence is
      expressing inability to answer the question due to lack of relevant information in
      the provided context as "supported_without_sentence". In cases where the
      sentence is making a general statement (e.g. outlining the steps to produce an answer, or
      summarizing previously stated sentences, or a transition sentence), use the
      string "general". In cases where the sentence is correctly stating a well-known fact,
      like a mathematical formula, use the string "well_known_fact". In cases where the
      sentence is performing numerical reasoning (e.g. addition, multiplication), use the
      string "numerical_reasoning".
      - fully_supported: a boolean indicating whether the sentence is fully supported by
      the documents.
      - This value should reflect the conclusion you drew at the end of your step-by-step
      breakdown in explanation.
      - If supporting_sentence_keys is an empty list, then fully_supported must be false.
      - Otherwise, use fully_supported to clarify whether everything in the response
      sentence is fully supported by the document text indicated in supporting_sentence_keys
      (fully_supported = true), or whether the sentence is only partially or incompletely
      supported by that document text (fully_supported = false).
      The all_utilized_sentence_keys field is a list of all sentences keys (e.g. '0a') that
      were used to construct the answer. Include every sentence that either directly supported
      the answer, or was implicitly used to construct the answer, even if it was not used
      in its entirety. Omit sentences that were not used and could have been removed from
      the documents without affecting the answer.
      You must respond with a valid JSON string. Use escapes for quotes, e.g. '\\\\"', and
      newlines, e.g. '\\\\n'. Do not write anything before or after the JSON string. Do not
      wrap the JSON string in backticks like '\\`' or '\\`json.
      As a reminder: your task is to review the response and assess which documents contain
      useful information pertaining to the question, and how each sentence in the response
      is supported by the text in the documents.
      """