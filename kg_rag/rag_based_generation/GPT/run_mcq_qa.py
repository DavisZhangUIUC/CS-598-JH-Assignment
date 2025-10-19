'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(
    config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(
    config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data[
    "SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(
    VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(
    SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


MODE = "3"
### MODE 0: Original KG_RAG                     ###
### MODE 1: jsonlize the context from KG search ###
### MODE 2: Add the prior domain knowledge      ###
### MODE 3: Combine MODE 1 & 2                  ###

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    question_df = question_df.iloc[191:]

    answer_list = []

    for index, row in tqdm(question_df.iterrows(), total=115):
        try:
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                                           QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Context: " + context + "\n" + "Question: " + question
                output = get_Gemini_response(
                    enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: jsonlize the context from KG search ###
                # Step 1: Retrieve unstructured context (same as MODE 0)
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                                           QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)

                # Step 2: Convert to structured JSON format
                structured_context = jsonlize_context(context)
                json_context = json.dumps(structured_context, indent=2)

                # Step 3: Create prompt with JSON context
                prompt = "Context (in structured JSON format): " + \
                    json_context + "\n\n" + "Question: " + question

                # Step 4: Get response from LLM
                output = get_Gemini_response(
                    prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "2":
                ### MODE 2: Add the prior domain knowledge      ###
                # Step 1: Retrieve unstructured context (same as MODE 0)
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                                           QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)

                # Step 2: Add prior domain knowledge
                prior_knowledge = (
                    "\n\nPrior Knowledge:\n"
                    "- Provenance & Symptoms information is useless\n"
                    "- Similar diseases tend to have similar gene associations"
                    "- HLA genes (HLA-A, HLA-B, HLA-C, HLA-DRB1, HLA-DQB1) are commonly associated with autoimmune and inflammatory diseases\n"
                    "- Genes in the same pathway or protein family often have similar disease associations\n"
                    "- Mendelian diseases typically have strong, specific gene associations\n"
                    "- Complex diseases often involve multiple genes with smaller effect sizes\n"
                    "- Genes expressed in the same tissue/organ are more likely to be associated with diseases of that organ\n"
                    "- Loss-of-function variants typically cause recessive diseases, gain-of-function variants cause dominant diseases\n"
                    "- Genes involved in DNA repair are associated with cancer predisposition syndromes\n"
                    "- Metabolic genes are often associated with inborn errors of metabolism\n"
                    "- When multiple diseases share a gene, consider the biological function of that gene."
                )
                enhanced_context = context + prior_knowledge

                # Step 3: Create enriched prompt
                prompt = "Context: " + enhanced_context + "\n\n" + "Question: " + question

                # Step 4: Get response from LLM
                output = get_Gemini_response(
                    prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "3":
                ### MODE 3: Combine MODE 1 & 2                  ###
                # Step 1: Retrieve unstructured context
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME,
                                           QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)

                # Step 2: Convert to structured JSON format
                structured_context = jsonlize_context(context)
                json_context = json.dumps(structured_context, indent=2)

                # Step 3: Add prior domain knowledge
                prior_knowledge = (
                    "\n\nPrior Domain Knowledge:\n"
                    "- Provenance & Symptoms information is useless\n"
                    "- Similar diseases tend to have similar gene associations\n"
                    "- HLA genes (HLA-A, HLA-B, HLA-C, HLA-DRB1, HLA-DQB1) are commonly associated with autoimmune and inflammatory diseases\n"
                    "- Genes in the same pathway or protein family often have similar disease associations\n"
                    "- Mendelian diseases typically have strong, specific gene associations\n"
                    "- Complex diseases often involve multiple genes with smaller effect sizes\n"
                    "- Genes expressed in the same tissue/organ are more likely to be associated with diseases of that organ\n"
                    "- Loss-of-function variants typically cause recessive diseases, gain-of-function variants cause dominant diseases\n"
                    "- Genes involved in DNA repair are associated with cancer predisposition syndromes\n"
                    "- Metabolic genes are often associated with inborn errors of metabolism\n"
                    "- When multiple diseases share a gene, consider the biological function of that gene."
                )

                # Step 4: Create prompt with both JSON context and prior knowledge
                prompt = "Context (in structured JSON format): " + json_context + \
                    prior_knowledge + "\n\n" + "Question: " + question
                
                print("Prompt: ", prompt)

                # Step 5: Get response from LLM
                output = get_Gemini_response(
                    prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            answer_list.append((row["text"], row["correct_node"], output))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))

    answer_df = pd.DataFrame(answer_list, columns=[
                             "question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=4),)
    answer_df.to_csv(output_file, index=False, header=True)
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))


if __name__ == "__main__":
    main()
