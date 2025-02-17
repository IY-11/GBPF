import attr
import spacy
from functools import partial
from nltk.corpus import wordnet as wn
from Attacker.get_NE_list import NE_list
import numpy as np
from config import args, config_pwws_use_NE, \
    config_data
from dataProcess import Tokenizer
from torch import nn
from tool import get_random, str2seq, strs2seq
from Attacker.typos import typos
import random
random.seed(667)

config_device = args.device
config_dataset = args.dataset
nlp = spacy.load('en_core_web_sm')
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

supported_pos_tags = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adverb, like "however wherever whenever"
]


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    try:
        sim = doc[start: original.i + window_size].similarity(synonym)
    except:
        synonym = nlp(synonym.text)
        sim = doc[start: original.i + window_size].similarity(synonym)
    return sim


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj


def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True

def get_similarity_words(word:str) -> str:
    word = nlp(word)[0]
    return {res.candidate_word.lower() for res in _generate_synonym_candidates(word, -1)}

class Adversary():
    """An Adversary tries to fool a model on a given example."""

    def __init__(self, net, max_perturbed_percent=0.25):
        self.net = net
        self.max_perturbed_percent = max_perturbed_percent

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.
        Args:
        model: a TextClassificationModel.
        dataset: a TextClassificationDataset.
        device: torch device.
        Returns: pair of
        - list of 0-1 adversarial loss of same length as |dataset|
        - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x

    def check_diff(self, sentence, perturbed_sentence):
        words = sentence.split()
        perturbed_words = perturbed_sentence.split()
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count

class GAAdversary(Adversary):
    """  GA attack method.  """

    def __init__(self, net, vocab, tokenizer, maxlen, iterations_num=20, pop_max_size=60):
        super(GAAdversary, self).__init__(net)
        self.max_iters = iterations_num
        self.pop_size = pop_max_size
        self.temp = 0.3
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def predict_batch(self, sentences):  # Done
        seqs = [" ".join(words) for words in sentences]
        tem = strs2seq(seqs, self.vocab, self.tokenizer, self.maxlen).to(config_device)
        tem, _  = self.net.predict_class(tem, True)
        tem = tem.cpu().numpy()
        # tem = self._softmax(tem)
        return tem

    def predict(self, sentence):  # Done
        tem = str2seq(" ".join(sentence), self.vocab, self.tokenizer, self.maxlen).to(config_device)
        tem, _ = self.net.predict_class(tem, True)
        # tem = self._softmax(tem)
        tem = tem[0].cpu().numpy()
        return tem

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_cur, x_orig, ori_label, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(new_x_list)

        new_x_scores = 1 - new_x_preds[:, ori_label]
        orig_score = 1 - self.predict(x_cur)[ori_label]
        new_x_scores = new_x_scores - orig_score

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur

    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, ori_label):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(np.array(x_orig) != np.array(x_cur)) < np.sum(
                np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, ori_label, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, ori_label, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, ori_label) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def check_return(self, perturbed_words, ori_words, ori_label):
        perturbed_text = " ".join(perturbed_words)
        clean_text = " ".join(ori_words)
        if self.check_diff(clean_text, perturbed_text) / len(ori_words) > self.max_perturbed_percent:
            return False, clean_text, ori_label
        else:
            # adv_label = self.target_model.query([perturbed_text], [ori_label])[1][0]
            perturbed_vec = str2seq(perturbed_text, self.vocab, self.tokenizer, self.maxlen).to(config_device)
            adv_label = self.net.predict_class(perturbed_vec)[0]
            # assert (adv_label != ori_label)
            return adv_label != ori_label, perturbed_text, adv_label

    def run(self, sentence, ori_label):

        # x_orig = np.array(sentence.split())
        x_orig = sentence.split()
        x_len = len(x_orig)

        neigbhours_list = []
        for i in range(x_len):
            # neigbhours_list.append(self.synonym_selector.find_synonyms(x_orig[i]))
            neigbhours_list.append(get_similarity_words(x_orig[i]))

        neighbours_len = [len(x) for x in neigbhours_list]
        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))

        if np.sum(w_select_probs) == 0:
            return False, sentence, ori_label

        w_select_probs = w_select_probs / np.sum(w_select_probs)

        pop = self.generate_population(
            x_orig, neigbhours_list, w_select_probs, ori_label, self.pop_size)
        for i in range(self.max_iters):
            pop_preds = self.predict_batch(pop)
            pop_scores = 1 - pop_preds[:, ori_label]
            # print('\t\t', i, ' -- ', np.max(pop_scores))
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if np.argmax(pop_preds[top_attack, :]) != ori_label:
                return self.check_return(pop[top_attack], x_orig, ori_label)
            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size - 1)]
            childs = [self.perturb(
                x, x_orig, neigbhours_list, w_select_probs, ori_label) for x in childs]
            pop = elite + childs

        return False, sentence, ori_label


def textfool_generate_synonym_candidates(doc, disambiguate=False, rank_fn=None):
    '''
    Generate synonym candidates.

    For each token in the doc, the list of WordNet synonyms is expanded.
    the synonyms are then ranked by their GloVe similarity to the original
    token and a context window around the token.

    :param disambiguate: Whether to use lesk sense disambiguation before
            expanding the synonyms.
    :param rank_fn: Functions that takes (doc, original_token, synonym) and
            returns a similarity score
    '''
    if rank_fn is None:
        rank_fn=vsm_similarity

    candidates = []
    for position, token in enumerate(doc):
        if token.tag_ in supported_pos_tags:
            wordnet_pos = _get_wordnet_pos(token)
            wordnet_synonyms = []
            if disambiguate:
                try:
                    synset = disambiguate(
                           doc.text, token.text, pos=wordnet_pos)
                    wordnet_synonyms = synset.lemmas()
                except:
                    continue
            else:
                synsets = wn.synsets(token.text, pos=wordnet_pos)
                for synset in synsets:
                    wordnet_synonyms.extend(synset.lemmas())

            synonyms = []
            for wordnet_synonym in wordnet_synonyms:
                spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
                synonyms.append(spacy_synonym)

            synonyms = filter(partial(_synonym_prefilter_fn, token),
                              synonyms)
            # synonyms = reversed(sorted(synonyms,
            #                     key=partial(rank_fn, doc, token)))

            for rank, synonym in enumerate(synonyms):
                candidate_word = synonym.text
                candidate = SubstitutionCandidate(
                        token_position=position,
                        similarity_rank=rank,
                        original_token=token,
                        candidate_word=candidate_word)
                candidates.append(candidate)

    return candidates


def _generate_synonym_candidates(token, token_position, rank_fn=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    if rank_fn is None:
        rank_fn = vsm_similarity
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            candidate = SubstitutionCandidate(
                token_position=token_position,
                similarity_rank=None,
                original_token=token,
                candidate_word=candidate_word)
            candidates.append(candidate)
    return candidates


def _compile_perturbed_tokens(doc, accepted_candidates):
    '''
    Traverse the list of accepted candidates and do the token substitutions.
    '''
    candidate_by_position = {}
    for candidate in accepted_candidates:
        candidate_by_position[candidate.token_position] = candidate

    final_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if position in candidate_by_position:
            candidate = candidate_by_position[position]
            word = candidate.candidate_word.replace('_', ' ')
        final_tokens.append(word)

    return final_tokens

def _generate_typo_candidates(doc, min_token_length=4, rank=1000):
    candidates = []
    for position, token in enumerate(doc):
        if (len(token)) < min_token_length:
            continue

        for typo in typos(token.text):
            candidate = SubstitutionCandidate(
                    token_position=position,
                    similarity_rank=rank,
                    original_token=token,
                    candidate_word=typo)
            candidates.append(candidate)

    return candidates


# def perturb_text(*args, **kwargs):
#     pass

def print_paraphrase(*args, **kwargs):
    pass

def get_pertubed_text(*args, **kwargs):
    pass


def random_attack(
        sentence,
        y_true,
        net,
        vocab,
        tokenizer,
        maxlen,
        verbose=False,
        sub_rate_limit=None):
    doc = nlp(sentence)
    if sub_rate_limit: sub_rate_limit = int(sub_rate_limit * len(doc))
    else: sub_rate_limit = len(doc)

    def halt_conditionh_func(perturbed_text):
        perturbed_vector = str2seq(perturbed_text, vocab, tokenizer, maxlen).to(config_device)
        predict = net.predict_class(perturbed_vector)[0]
        return predict != y_true


    candidates_list = []
    for idx, token in enumerate(doc):
        if idx >= maxlen: break
        candidates = _generate_synonym_candidates(token=token, token_position=idx, rank_fn=None)
        if len(candidates) > 0: candidates_list.append((idx, candidates))

    upper = min(len(candidates_list), sub_rate_limit)
    lower = upper // 3
    sub_num = get_random(lower, upper)
    sub_pos = random.sample(candidates_list, sub_num)
    change_tuple_list = []

    accepted_candidates = []



    for token_pos, candidates in sub_pos:
        substitution = random.sample(candidates, 1)[0]
        accepted_candidates.append(substitution)
        change_tuple_list.append((token_pos, substitution.original_token, substitution.candidate_word, None, 'tagNONE'))
        perturbed_text = ' '.join(
            _compile_perturbed_tokens(doc, accepted_candidates))
        if halt_conditionh_func(perturbed_text): break
        if verbose:
            print(f'origin token pos {token_pos}, origin token {substitution.original_token}, candidate token {substitution.candidate_word}')

    perturbed_text = ' '.join(
        _compile_perturbed_tokens(doc, accepted_candidates))

    sub_rate = len(change_tuple_list) / len(doc)
    ne_rate = 0.0
    adv_vec = str2seq(perturbed_text, vocab, tokenizer, maxlen).to(config_device)
    adv_y = net.predict_class(adv_vec)[0]

    return perturbed_text, adv_y, sub_rate, ne_rate, change_tuple_list


def textfool_perturb_text(
        sentence,
        net,
        vocab,
        tokenizer,
        maxlen,
        true_y,
        use_typos=False,
        rank_fn=None,
        heuristic_fn=None,
        verbose=False,
        sub_rate_limit=None):
    '''
    Perturb the text by replacing some words with their WordNet synonyms,
    sorting by GloVe similarity between the synonym and the original context
    window, and optional heuristic.

    :param doc: Document to perturb.
    :type doc: spacy.tokens.doc.Doc
    :param rank_fn: See `_generate_synonym_candidates``.
    :param heuristic_fn: Ranks the best synonyms using the heuristic.
            If the value of the heuristic is negative, the candidate
            substitution is rejected.
    :param halt_condition_fn: Returns true when the perturbation is
            satisfactory enough.
    :param verbose:
    ATTENTION:
    This function is originally from https://github.com/bogdan-kulynych/textfool
    '''
    
    def halt_conditionh_func(perturbed_text):
        perturbed_vector = str2seq(perturbed_text, vocab, tokenizer, maxlen).to(config_device)
        predict = net.predict_class(perturbed_vector)[0]
        return predict != true_y


    doc = nlp(sentence)
    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    candidates = textfool_generate_synonym_candidates(doc, rank_fn=rank_fn)
    if use_typos:
        candidates.extend(_generate_typo_candidates(doc))

    if sub_rate_limit: sub_rate_limit = int(sub_rate_limit * len(doc))
    else: sub_rate_limit = len(doc)

    perturbed_positions = set()
    accepted_candidates = []
    perturbed_text = doc.text
    if verbose:
        print('Got {} candidates'.format(len(candidates)))

    sorted_candidates = zip(
            map(partial(heuristic_fn, perturbed_text), candidates),
            candidates)
    sorted_candidates = list(sorted(sorted_candidates,
            key=lambda t: t[0]))
    change_tuple_list = []

    while len(sorted_candidates) > 0 and len(change_tuple_list) < sub_rate_limit:
        score, candidate = sorted_candidates.pop()
        if score < 0: continue
        if candidate.token_position >= maxlen: break
        if candidate.token_position not in perturbed_positions:
            perturbed_positions.add(candidate.token_position)
            accepted_candidates.append(candidate)
            if verbose:
                print('Candidate:', candidate)
                print('Candidate score:', heuristic_fn(perturbed_text, candidate))
                print('Candidate accepted.')

            perturbed_text = ' '.join(_compile_perturbed_tokens(doc, accepted_candidates))

            change_tuple_list.append((candidate.token_position, candidate.original_token, candidate.candidate_word, score, 'TAGNONE'))
            if halt_conditionh_func(perturbed_text):
                break
            if len(sorted_candidates) > 0:
                _, candidates = zip(*sorted_candidates)
                sorted_candidates = zip(
                        map(partial(heuristic_fn, perturbed_text),
                            candidates),
                        candidates)
                sorted_candidates = list(sorted(sorted_candidates,
                        key=lambda t: t[0]))
    sub_rate = len(change_tuple_list) / len(doc)
    ne_rate = 0.0
    adv_vec = str2seq(perturbed_text, vocab, tokenizer, maxlen).to(config_device)
    adv_y = net.predict_class(adv_vec)[0]
    return perturbed_text, adv_y, sub_rate, ne_rate, change_tuple_list


'''
    ATTENTION:
    Below three functions (PWWS, evaluate_word_saliency, adversarial_paraphrase)
    is an nonofficial PyTorch version of https://github.com/JHL-HUST/PWWS
'''

def PWWS(
        doc,
        true_y,
        word_saliency_list=None,
        rank_fn=None,
        heuristic_fn=None,  # Defined in adversarial_tools.py
        halt_condition_fn=None,  # Defined in adversarial_tools.py
        verbose=True,
        sub_rate_limit=None):


    # defined in Eq.(8)
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)
    perturbed_doc = doc
    perturbed_text = perturbed_doc.text

    substitute_count = 0  # calculate how many substitutions used in a doc
    substitute_tuple_list = []  # save the information of substitute word

    word_saliency_array = np.array([word_tuple[2] for word_tuple in word_saliency_list])
    word_saliency_array = softmax(word_saliency_array)

    NE_candidates = NE_list.L[config_dataset][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = config_pwws_use_NE  # whether use NE as a substitute

    max_len = config_data[config_dataset].padding_maxlen

    if sub_rate_limit: sub_rate_limit = int(sub_rate_limit * len(doc))
    else: sub_rate_limit = len(doc)

    # for each word w_i in x, use WordNet to build a synonym set L_i
    for (position, token, word_saliency, tag) in word_saliency_list:
        if position >= max_len: break


        candidates = []
        if use_NE:
            NER_tag = token.ent_type_
            if NER_tag in NE_tags:
                candidate = SubstitutionCandidate(position, 0, token, NE_candidates[NER_tag])
                candidates.append(candidate)
            else:
                candidates = _generate_synonym_candidates(token=token, token_position=position, rank_fn=rank_fn)
        else:
            candidates = _generate_synonym_candidates(token=token, token_position=position, rank_fn=rank_fn)

        if len(candidates) == 0: continue

        # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
        sorted_candidates = zip(map(partial(heuristic_fn, doc.text), candidates), candidates)
        # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
        sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

        # delta_p_star is defined in Eq.(5); substitute is w_i^*
        delta_p_star, substitute = sorted_candidates.pop()

        # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
        substitute_tuple_list.append(
            (position, token.text, substitute, delta_p_star * word_saliency_array[position], token.tag_))

    # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
    sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    NE_count = 0  # calculate how many NE used in a doc
    change_tuple_list = []
    for (position, token, substitute, score, tag) in sorted_substitute_tuple_list:
        if len(change_tuple_list) > sub_rate_limit: break
        # if score <= 0:
        #     break
        if nlp(token)[0].ent_type_ in NE_tags:
            NE_count += 1
        change_tuple_list.append((position, token, substitute, score, tag))
        perturbed_text = ' '.join(_compile_perturbed_tokens(perturbed_doc, [substitute]))
        perturbed_doc = nlp(perturbed_text)
        substitute_count += 1
        if halt_condition_fn(perturbed_text):
            if verbose:
                print("use", substitute_count, "substitution; use", NE_count, 'NE')
            sub_rate = substitute_count / len(doc)
            if substitute_count == 0: NE_rate = 0.0
            else: NE_rate = NE_count / substitute_count
            return perturbed_text, sub_rate, NE_rate, change_tuple_list

    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')
    sub_rate = substitute_count / len(doc)
    if substitute_count == 0: NE_rate = 0.0
    else: NE_rate = NE_count / substitute_count

    return perturbed_text, sub_rate, NE_rate, change_tuple_list


def evaluate_word_saliency(doc, origin_vector, input_y, net):
    word_saliency_list = []

    # zero the code of the current word and calculate the amount of change in the classification probability
    max_len = config_data[config_dataset].padding_maxlen
    origin_prob = net.predict_prob(origin_vector, input_y)[0]
    for position in range(len(doc)):
        if position >= max_len:
            break
        without_word_vector = origin_vector.clone().detach().to(config_device)
        without_word_vector[position] = 0
        prob_without_word = net.predict_prob(without_word_vector, input_y)[0]

        # calculate S(x,w_i) defined in Eq.(6)
        word_saliency = origin_prob - prob_without_word
        word_saliency_list.append((position, doc[position], word_saliency, doc[position].tag_))


    position_word_list = []
    for word in word_saliency_list:
        position_word_list.append((word[0], word[1]))

    return position_word_list, word_saliency_list


def adversarial_paraphrase(input_text, origin_vector, true_y, tokenizer:Tokenizer,
                           vocab, net:nn.Module, verbose=False, sub_rate_limit=None):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text

    : generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        maxlen = config_data[config_dataset].padding_maxlen
        perturbed_vector = str2seq(perturbed_text, vocab, tokenizer, maxlen).to(config_device)
        predict = net.predict_class(perturbed_vector)[0]
        return predict != true_y

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        doc = nlp(text)
        maxlen = config_data[config_dataset].padding_maxlen
        perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
        perturbed_doc = ' '.join(perturbed_tokens)
        perturbed_vector = str2seq(perturbed_doc, vocab, tokenizer, maxlen).to(config_device)
        adv_y = net.predict_prob(perturbed_vector, true_y)[0]
        ori_y = net.predict_prob(origin_vector, true_y)[0]


        return ori_y - adv_y

    doc = nlp(input_text)

    # PWWS
    position_word_list, word_saliency_list = evaluate_word_saliency(doc, origin_vector, true_y, net)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(doc,
                                                                true_y,
                                                                word_saliency_list=word_saliency_list,
                                                                heuristic_fn=heuristic_fn,
                                                                halt_condition_fn=halt_condition_fn,
                                                                verbose=verbose,
                                                                sub_rate_limit=sub_rate_limit)

    # print("perturbed_text after perturb_text:", perturbed_text)

    maxlen = config_data[config_dataset].padding_maxlen
    perturbed_vector = str2seq(perturbed_text, vocab, tokenizer, maxlen).to(config_device)
    perturbed_y = net.predict_class(perturbed_vector)[0]
    if verbose:
        origin_prob = net.predict_prob(origin_vector, true_y)[0]
        perturbed_prob = net.predict_prob(perturbed_vector, true_y)[0]
        raw_score = origin_prob - perturbed_prob
        print('Prob before: ', origin_prob, '. Prob after: ', perturbed_prob,
              '. Prob shift: ', raw_score)
    return perturbed_text, perturbed_y, sub_rate, NE_rate, change_tuple_list






if __name__ == '__main__':
    text = "I went to an advance screening of this movie thinking I was about to embark on 120 minutes of cheezy lines, mindless plot, and the kind of nauseous acting that made The Postman one of the most malignant displays of cinematic blundering of our time. But I was shocked. Shocked to find a film starring Costner that appealed to the soul of the audience. Shocked that Ashton Kutcher could act in such a serious role. Shocked that a film starring both actually engaged and captured my own emotions. Not since 'Robin Hood' have I seen this Costner: full of depth and complex emotion. Kutcher seems to have tweaked the serious acting he played with in Butterfly Effect. These two actors came into this film with a serious, focused attitude that shone through in what I thought was one of the best films I've seen this year. No, its not an Oscar worthy movie. It's not an epic, or a profound social commentary film. Rather, its a story about a simple topic, illuminated in a way that brings that audience to a higher level of empathy than thought possible. That's what I think good film-making is and I for one am throughly impressed by this work. Bravo!"
    # doc = nlp(text)
    # print(textfool_perturb_text(doc, use_typos=False, verbose=False))
    # print(_generate_synonym_candidates(doc[10], 10))
    print(random_attack(text, None, None, None, 300, verbose=True))