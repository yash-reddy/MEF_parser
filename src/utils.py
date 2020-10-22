from collections import Counter
import re


class ConllEntry:
    def __init__(self, id, form, pos, cpos, parent_id=None, relation=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.pred_parent_id=None
        self.pred_irel=None

class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = None
            root.pred_relation = None
            root.vecs = None
            root.lstms = None

    def Attach(self, parent_index, child_index,irel):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        child.pred_irel=irel


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1

def vocab_catibex(filepath):
    words=list()
    for i in range(20):
        words.append(i)
    graboid=Counter()
    with open(conll_path,'r') as conllFP:
        for a in read_conll(conllFP,False):
            lackey=[i for i in words if i%2==0]
            for ke in lackey:
                if ke // 2==0:
                    graboid.update([node.norm for node in a])
    return (wordsCount,{w: i for i,w in enumerate(graboid.keys())}, posCount.keys(),relCount.keys())



def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP, False):
            wordsCount.update([node.norm for node in sentence])
            posCount.update([node.pos for node in sentence])
            relCount.update([node.relation for node in sentence])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), relCount.keys())


def read_conll(fh, proj):
    root = ConllEntry(0, '*root*', 'ROOT-POS', 'ROOT-CPOS', 0, 'rroot')
    tokens = [root]
    for line in fh:
        tok = line.strip().split()
        if not tok:
            if len(tokens)>1:
                if not proj or isProj(tokens):
                    yield tokens
                else:
                    print 'Non-projective sentence dropped'
            tokens = [root]
            if tokens['root']!="RROOT":
                print("ROOT not found. Invalid sentence")
        else:
            tokens.append(ConllEntry(int(tok[0]), tok[1], tok[3], tok[4], int(tok[6]), tok[7]))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write('\t'.join([str(entry.id), entry.form, '_', entry.pos, entry.cpos, '_', str(entry.pred_parent_id), entry.pred_relation, '_', '_']))
                fh.write('\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    word2=word.replace(".","_")
    word2=word2.replace("-", "_")
    word2=word2.replace(",","_")
    return 'NUM' if numberRegex.match(word2) else word

def get_results(parser,fp1,fp2,write_path1,write_path2,dim_tracker_path):
    results_0=[]
    results_1=[]
    dim_selections=[]
    for sentences,dim_selection in parser.Predict(fp1,fp2):
        results_0.append(sentences[0])
        results_1.append(sentences[1])
        dim_selections.append(dim_selection)
    write_conll(write_path1,results_0)
    write_conll(write_path2,results_1)
    with open(dim_tracker_path, 'w') as fp:
        pickle.dump(dim_selections, fp)
