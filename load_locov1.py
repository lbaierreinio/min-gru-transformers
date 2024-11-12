# Most of the code in this file is adapted from https://github.com/HazyResearch/m2
# which is licensed under the Apache License, Version 2.0

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import InputExample
from tqdm import tqdm
import numpy as np
import random

# List of the 10 datasets in LoCoV1
# Values represent (dataset_name, split, document_column, query_column, subset)
LOCOV1_DATASETS = {
    "tau_scrolls_summ_screen_fd_config" : ("tau/scrolls", "train", "input", "output", "summ_screen_fd"),
    "tau_scrolls_gov_report_config" : ("tau/scrolls", "train", "input", "output", "gov_report"),
    "tau_scrolls_qmsum_config" : ("tau/scrolls", "train", "input", "output", "qmsum"),
    "qasper_title_config" : ("qasper", "train", "full_text", "title", None),
    "qasper_abstract_config" : ("qasper", "train", "full_text", "abstract", None),
    "multifieldqa_en_config" : ("long_bench", "train", "context", "input", "multifieldqa_en"),
    "wikimqa_config" : ("long_bench", "train", "context", "input", "2wikimqa"),
    "passage_retrieval_en_config" : ("long_bench", "train", "context", "input", "passage_retrieval_en"),
    "legal_case_reports" : ("legal_case_reports", "train", None, None, None),
    "courtlistener_html" : ("courtlistener", "train", "Document_HTML", "Query", "Document_HTML"),
    "courtlistener_plain_text" : ("courtlistener", "train", "Document_Plain_Text", "Query", "Document_Plain_Text"),
    "stackoverflow" : ("stackoverflow", "train", "passage", "query", None),
}


def gather_strong_negatives(query, relevant_documents, bm25_index, document_set, threshold_for_negatives):
    """Select a hard negative example for a query based on BM25 relevance results.
    
    Returns:
    - negative_selected (str): A randomly selected strong negative document
    """
    top_documents = bm25_index.get_top_n(query.split(), document_set, n=threshold_for_negatives+10000)
    strong_negatives = [doc for doc in top_documents if doc not in relevant_documents]
    strong_negatives = strong_negatives[:threshold_for_negatives] # Cutoff for negatives

    for relevant_doc in relevant_documents:
        assert relevant_doc in top_documents

    negative_selected = random.choice(strong_negatives)
    assert type(negative_selected) == str
    return negative_selected


def load_loco_from_hf(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):
    """
    Load the specified subset of the LoCoV1 dataset from HuggingFace.

    Returns:
    - corpus (dict): A dictionary from pid (str) to a passage (dict of the form {"title": "", "text": ...})
    - queries (dict): A dictionary from qid (str) to query text (str)
    - qrels (dict): A dictionary from qid (str) to relevant passages (dict) with pids as keys and values set to 1 (relevance score of 1)
    """
    # Only load relevant data from loco, based on input configs
    if "qasper" == dataset_name:
        if "abstract" == query_column:
            dataset_choice = "qasper_abstract"
            split = "test"
        elif "title" == query_column:
            dataset_choice = "qasper_title"
            split = "test"
        else:
            raise ValueError("No dataset specified for QASPER!")
    elif type(subset) == str and "passage_retrieval" in subset:
        dataset_choice = "passage_retrieval"
        split = "test"
    elif "legal_case_reports" == dataset_name:
        dataset_choice = dataset_name
        split = "test"
    elif "courtlistener" == dataset_name:
        if "Document_HTML" == subset:
            dataset_choice = "courtlistener_HTML"
            split = "test"
        elif "Document_Plain_Text" == subset:
            dataset_choice = "courtlistener_Plain_Text"
            split = "test"
    elif "stackoverflow" == dataset_name:
        dataset_choice = dataset_name
        split = "test"
    elif "multifieldqa_en" == subset:
        dataset_choice = "multifieldqa"
        split = "test"
    else:
        dataset_choice = subset
        split = "test"

    if split == "validation":
        split = "test"

    def filter_condition(example):
        return example['dataset'] == dataset_choice
    queries_dataset = load_dataset("hazyresearch/LoCoV1-Queries")[split].filter(filter_condition)
    documents = load_dataset("hazyresearch/LoCoV1-Documents")[split].filter(filter_condition)

    # Gather data into desired format
    queries = {}
    qrels = {}
    for row in tqdm(range(len(queries_dataset))):
        queries[queries_dataset[row]["qid"]] = queries_dataset[row]["query"]
        qrels_list = {}
        assert type(queries_dataset[row]["answer_pids"]) == list
        for pid in queries_dataset[row]["answer_pids"]:
            qrels_list[pid] = 1
        qrels[queries_dataset[row]["qid"]] = qrels_list
    
    corpus = {}
    for row in tqdm(range(len(documents))):
        corpus[documents[row]['pid']] = {"title": "", "text": documents[row]["passage"]}

    if "qasper" in dataset_choice:
        queries = {key: value for key, value in queries.items() if corpus[key.replace("Query", "Passage")]['text'] is not None} # Check to make sure corpus passage is not None
        corpus = {key: value for key, value in corpus.items() if corpus[key.replace("Query", "Passage")]['text'] is not None} # Check to make sure corpus passage is not None

    print("Example Query")
    print(list(queries.values())[5])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[5]['text'][:200])

    return corpus, queries, qrels


def collect_dataset(dataset):
    """
    A thin wrapper around load_loco_from_hf that just prints additional information about
    the query and passage lengths of the specified dataset.
    
    Returns:
    - corpus (dict): A dictionary from pid (str) to a passage (dict of the form {"title": "", "text": ...})
    - queries (dict): A dictionary from qid (str) to query text (str)
    - qrels (dict): A dictionary from qid (str) to relevant passages (dict) with pids as keys and values set to 1 (relevance score of 1)
    """
    dataset_name, split, document_column, query_column, subset = dataset
    corpus, queries, qrels = load_loco_from_hf(dataset_name, split, document_column, query_column, subset)

    print("-----------------------------------------------")
    print("Dataset: " + str(subset))
    query_lengths = [len(query) for query in list(queries.values())]
    print("Query Lengths - 25, 50, 75, and 100 Percentiles:")
    print(np.percentile(query_lengths, [25, 50, 75, 100]))
    doc_lengths = [len(doc['text']) for doc in list(corpus.values())]
    print("Document Lengths - 25, 50, 75, and 100 Percentiles:")
    print(np.percentile(doc_lengths, [25, 50, 75, 100]))
    print("-----------------------------------------------")
    
    return corpus, queries, qrels


def make_input_examples(query, positive_passage, negative_passage, loss_choice):
    """
    Given a data point (query, positive passage, negative passage), format it
    in a way that is compatible with the specified loss function.

    See https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html

    Returns:
    - input_examples (list): A list of InputExample objects
    """

    input_examples = []

    if loss_choice in ["multiple_negatives_ranking_loss", "triplet_loss", "assisted_embedding_loss"]:
        input_examples.append(InputExample(texts=[query, positive_passage, negative_passage]))
    elif loss_choice in ["contrastive_loss", "online_contrastive_loss"]:
        input_examples.append(InputExample(texts=[query, positive_passage], label=1))
        input_examples.append(InputExample(texts=[query, negative_passage], label=0))
    elif loss_choice in ["cosine_similarity_loss"]:
        input_examples.append(InputExample(texts=[query, positive_passage], label=1.0))
        input_examples.append(InputExample(texts=[query, negative_passage], label=0.0))
    elif loss_choice in ["mega_batch_margin_loss"]:
        input_examples.append(InputExample(texts=[query, positive_passage]))
    else:
        raise ValueError("Invalid loss function!")
    
    return input_examples


def gather_loco_training_examples(
        loco_example_count, loco_evaluation_set_count, threshold_for_negatives, negatives_per_query,
        loss_choice, use_negatives_from_same_dataset_for_multidataset_finetuning
    ):
    """
    Gather training and validation examples from the LoCoV1 dataset for model training.

    The LoCoV1 dataset contains various subsets of data pertaining to different tasks and contexts.
    This function assembles a dataset of positive and negative examples from the various LoCoV1 subsets,
    and stores these examples in formats compatible with multiple loss functions.
    Additionally, the function can store negative samples in a memory bank for future reference if specified.

    Args:
    - loco_example_count (int): Number of training examples to gather from LoCoV1.
    - loco_evaluation_set_count (int): Number of evaluation examples to gather from LoCoV1.
    - threshold_for_negatives (float): Relevance threshold to use for selecting strong negatives.
    - negatives_per_query (int): Number of negative examples per query.
    - use_negatives_from_same_dataset_for_MNRL (bool): Flag to use negatives from the same dataset in multiple negatives ranking loss.
    - loss_choice (str): The loss function for which to generate examples (e.g., "contrastive_loss").
    - use_negatives_from_same_dataset_for_multidataset_finetuning (bool): If True, includes negatives from the same dataset for multi-dataset fine-tuning (default is False).

    Returns:
    - long_context_training_examples (list): A list of training examples in InputExample format, containing positive and negative pairs.
    - long_context_validation_examples (list): A list of validation examples in InputExample format, containing positive and negative pairs.
    """

    training_datasets = list(LOCOV1_DATASETS.values())
    long_context_training_examples = []
    long_context_validation_examples = []

    # Build data for each dataset
    for dataset in training_datasets:
        print(f"Collecting training examples from {dataset[0]}_{dataset[4]}_{dataset[3]}!")

        # Create set of negatives across all passages
        if not use_negatives_from_same_dataset_for_multidataset_finetuning:
            total_corpus_passages = []
            for training_dataset in training_datasets:
                if training_dataset != dataset:
                    corpus, queries, qrels = collect_dataset(training_dataset)
                    for key in corpus.keys():
                        total_corpus_passages.append(corpus[key]['text'])

        corpus, queries, qrels = collect_dataset(dataset)
        
        total_corpus_keys = list(corpus.keys())

        document_set = [corpus[key]['text'] for key in total_corpus_keys]
        tokenized_documents = [doc.split() for doc in document_set]
        
        bm25_index = BM25Okapi(tokenized_documents)

        # Edge case: less documents than the negatives count per query
        num_negative_queries = negatives_per_query
        if len(total_corpus_keys) <= 32:
            num_negative_queries = len(total_corpus_keys) - 1

        for i, query_key in enumerate(tqdm(queries)):
            query = queries[query_key]
            assert type(query) == str
            positive_passage_keys = list(qrels[query_key].keys())

            used_negative_keys = set()

            # Get negatives_per_query negative examples for each positive passage
            for _ in range(num_negative_queries):
                for pid in positive_passage_keys:
                    positive_passage = corpus[pid]['text']
                    assert type(positive_passage) == str
                    
                    # Choose a negative passage
                    if random.choice([0, 1]) == 1 and not use_negatives_from_same_dataset_for_multidataset_finetuning:
                        # Get a random one
                        negative_passage = random.choice(total_corpus_passages)
                    else:
                        if threshold_for_negatives < 0:
                            random_negative_passage_key = random.choice(total_corpus_keys)
                            while random_negative_passage_key in positive_passage_keys or random_negative_passage_key in used_negative_keys:
                                random_negative_passage_key = random.choice(total_corpus_keys)
                            negative_passage = corpus[random_negative_passage_key]['text']
                            used_negative_keys.add(random_negative_passage_key)
                            assert type(negative_passage) == str
                        # Or use BM25 to get a strong negative passage
                        else:
                            relevant_documents = [corpus[key]['text'] for key in positive_passage_keys]
                            negative_passage = gather_strong_negatives(query, relevant_documents, bm25_index, document_set, threshold_for_negatives)
                    
                    # 90% training, 10% val split
                    if i % 10 == 0:
                        long_context_validation_examples.extend(
                            make_input_examples(query, positive_passage, negative_passage, loss_choice)
                        )
                    else:
                        long_context_training_examples.extend(
                            make_input_examples(query, positive_passage, negative_passage, loss_choice)
                        )

    print("Completed creating datasets")

    return long_context_training_examples, long_context_validation_examples

if __name__ == "__main__":
    # Reference usage
    # NOTE: turning on hard negative mining is slow
    loco_example_count = 10000000 #500000 #250000
    loco_evaluation_set_count = 2000 # args.loco_evaluation_set_count
    threshold_for_negatives = -1 #-1 indicates negatives are randomly sampled, negative passages are randomly sampled from positive number upwards
    negatives_per_query = 32 # args.negatives_per_query # Number of negatives to add per query-positive passage pair
    use_negatives_from_same_dataset_for_MNRL = False
    use_memory_bank = True
    loss_choice = "multiple_negatives_ranking_loss" # args.loss_choice # Options: "orthogonal_projection_loss" #"online_contrastive_loss" #"triplet_loss" #"contrastive_loss" "multiple_negatives_ranking_loss"
    use_negatives_from_same_dataset_for_multidataset_finetuning = True
    long_context_training_examples, long_context_validation_examples = gather_loco_training_examples(
       loco_example_count, loco_evaluation_set_count, threshold_for_negatives,
        negatives_per_query, use_negatives_from_same_dataset_for_MNRL,
        use_memory_bank, loss_choice,
        use_negatives_from_same_dataset_for_multidataset_finetuning
    )
    from sentence_transformers import datasets
    train_dataloader = datasets.NoDuplicatesDataLoader(long_context_training_examples, batch_size=32)
    # Each batch will be a list of sentence_transformers.readers.InputExample.InputExample objects
