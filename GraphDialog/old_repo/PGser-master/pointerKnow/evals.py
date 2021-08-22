import torch
import torch.nn.functional as F
import random
import config
import numpy as np

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):
    """
    distinct
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

def rebuild_sentence(sent, vocab, oovs):
    words = []
    for token in sent:
        if token == config.EOS_token:
            break
        elif token == config.PAD_token:
            continue;
        elif token in vocab.index2word:
            words.append(vocab.index2word[token])
        else:
            words.append(oovs[token-vocab.num_words])
    return words

def evaluate_generation(generator,
                        data_iter,
                        vocab,
                        save_file=None):
    """
    evaluate_generation
    """
    refs = []
    hyps = []

    for data in data_iter:
        tokens, scores, batch_oovs = generator(data)
        truths = data[-1].transpose(0,1).tolist()
        for i,sent in enumerate(tokens):
            prediction = rebuild_sentence(sent,vocab,batch_oovs[i])
            truth = rebuild_sentence(truths[i],vocab,batch_oovs[i])
            refs.append(truth)
            hyps.append(prediction)

    report_message = []

    avg_len = np.average([len(s) for s in hyps])
    report_message.append("Avg_Len-{:.3f}".format(avg_len))

    bleu_1, bleu_2 = bleu(hyps, refs)
    report_message.append("Bleu-{:.4f}/{:.4f}".format(bleu_1, bleu_2))

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(hyps)
    report_message.append("Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2))

    report_message = "   ".join(report_message)

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(refs)
    avg_len = np.average([len(s) for s in refs])
    target_message = "Target:   AVG_LEN-{:.3f}   ".format(avg_len) + \
        "Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2)

    message = report_message + "\n" + target_message

    if save_file is not None:
        write_results(refs, hyps, save_file)
        print("Saved generation results to '{}'".format(save_file))
    print(message)


def write_results(refs, hyps, results_file):
    """
    write_results
    """
    with open(results_file, "w", encoding="utf-8") as f:
        for truth, prediction in zip(refs, hyps):
            r = 'truth: '+' '.join(truth)+'\nprediction: '+' '.join(prediction)
            f.write("{}\n".format(r))




# def evaluateInput(encoder, sec_encoder, decoder, searcher, voc, wiki_strings):
#     input_sentence = ''
#     while(1):
#         try:
#             doc_idx = int(input('document index:'))
#             sec_idx = int(input('section index:'))
#             sec_sentence = wiki_strings[doc_idx][sec_idx]
#             # Get input sentence
#             input_sentence = input('> ')
#             # Check if it is quit case
#             if input_sentence == 'q' or input_sentence == 'quit': break
#             # Normalize sentence
#             input_sentence = normalizeString(input_sentence)
#             # Evaluate sentence
#             output_words = evaluate(encoder, sec_encoder, decoder, searcher, voc, input_sentence, sec_sentence)
#             # Format and print response sentence
#             output_words[:] = [x for x in output_words ]
#             print('Bot:', ' '.join(output_words))

#         except KeyError:
#             print("Error: Encountered unknown word.")
