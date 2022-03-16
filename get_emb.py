from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from matscholar import MatScholarProcess
import json
from operator import itemgetter
from scipy import stats


def process_file(text):
    zt_answer = {}
    for i, z in enumerate(text):
        z_split = z.split(', ')
        if z_split[1] != 'NaN':
            mat_name = z_split[0][1:-1]
            mat_rec = float(z_split[1])
            if mat_name in zt_answer.keys():
                if mat_rec > zt_answer[mat_name]:
                    zt_answer[mat_name] = mat_rec
            else:
                zt_answer[mat_name] = mat_rec

    sorted_zt = sorted(zt_answer.items(), key=lambda item: item[1], reverse=True)
    zt_rank = {}
    rank = 1
    processor = MatScholarProcess()
    for s in sorted_zt:
        processed, _ = processor.process(s[0])
        if processed[0] not in zt_rank.keys():
            zt_rank[processed[0]] = rank
            rank += 1
    return zt_rank


# word embedding
w2v_model = Word2Vec.load("models/pretrained_embeddings")
word_vectors = w2v_model.wv
print(len(word_vectors.vocab))
sims = word_vectors.most_similar('thermoelectric', topn=529686)
'''
# output embedding
model = Word2Vec.load("models/pretrained_embeddings")
outv = KeyedVectors(529686)
outv.vocab = model.wv.vocab
outv.index2word = model.wv.index2word
outv.syn0 = model.syn1neg
sims = outv.most_similar(positive=[model.syn1neg[model.wv.vocab['thermoelectric'].index]], topn=529686)
'''
with open('zt.json', 'r', encoding='utf-8') as f:
    zt = json.load(f)
zt_split = zt[2:-2].split('], [')
zt_rank = process_file(zt_split)
# 194 in zt: # 1 NaPb47Sr2Te50 2.27156 # 2 Na2Pb98Se15Te85 1.703161584 # 3 Na2Pb98Se25Te75 1.636067789
with open('PF.txt', 'r', encoding='utf-8') as f:
    pf = str(f.readlines())
pf_split = pf[4:-4].split('], [')
pf_rank = process_file(pf_split)
# 281 in max power factor: # 1 Bi2Te3 0.006728 # 2 Co2NaO4 0.0066667 # 3 HfNi4Sn4Ti2Zr 0.0065971

sim_dict = {}
rank = 1
for s in sims:
    # if s[0] in pf_rank.keys():
    if s[0] in zt_rank.keys():
        sim_dict[s[0]] = {}
        sim_dict[s[0]]['rank'] = rank
        sim_dict[s[0]]['value'] = s[1]
        rank += 1
# print(sim_dict)
# output embedding: 1 Bi2Te3 2 Ca9Mn10NdO30 3 Ca50Mn49NbO150
# word embedding: 1 Bi2Te3 2 Ca50Mn49NbO150 3 CoFe3LaSb12

compare_ex = []
sim_rank = []
rank = 1
for r in zt_rank.keys():
# for r in pf_rank.keys():
    if r in sim_dict.keys():
        compare_ex.append(rank)
        sim_rank.append(sim_dict[r]['rank'])
        rank += 1
ex, L_s = zip(*sorted(enumerate(sim_rank), key=itemgetter(1)))
print(len(sim_rank))
print(stats.spearmanr(compare_ex, ex))
# output embedding: zt: 0.32, mpf: 0.53
# word embedding: zt: 0.41, mpf: 0.58
