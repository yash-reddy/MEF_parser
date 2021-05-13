from dynet import *
from utils import ParseForest, read_conll, write_conll
import utils, time, random, math
import numpy as np
import pickle


class EasyFirstLSTM:
	def __init__(self, words, pos1, pos2, rels1, rels2, w2i, options):

		self.embed_mode=options.embed_mode
		self.rep_mode=options.rep_mode
		self.blstm_mode=options.blstm_mode
		self.ul_mode=options.ul_mode
		self.parse_mode=options.parse_mode
		self.priority_dim=options.priority_dim

		random.seed(1)
		self.model = ParameterCollection()
		self.trainer = AdamTrainer(self.model)

		self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
		self.activation = self.activations[options.activation]

		self.k = options.window
		self.ldims = options.lstm_dims
		self.wdims = options.wembedding_dims
		self.pdims = options.pembedding_dims
		self.rdims = options.rembedding_dims
		self.oracle = options.oracle
		self.layers = options.lstm_layers
		self.wordsCount = words
		self.vocab = {word: ind+3 for word, ind in w2i.items()}
		self.pos = [{word: ind+3 for ind, word in enumerate(pos1)},{word: ind+3 for ind, word in enumerate(pos2)}]
		self.pos1 = {word: ind+3 for ind, word in enumerate(pos1)}
		self.pos2 = {word: ind+3 for ind, word in enumerate(pos2)}
		self.rels1 = {word: ind for ind, word in enumerate(rels1)}
		self.rels2 = {word: ind for ind, word in enumerate(rels2)}
		self.irels = [rels1,rels2]


		self.hidden_units = options.hidden_units
		self.hidden2_units = options.hidden2_units

		self.external_embedding = None
		if options.external_embedding is not None:
			external_embedding_fp = open(options.external_embedding,'r')
			external_embedding_fp.readline()
			self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
			external_embedding_fp.close()
			self.edim = len(self.external_embedding.values()[0])
			self.noextrn = [0.0 for _ in range(self.edim)]
			self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
			self.elookup=self.model.add_lookup_parameters("extrn-lookup", (len(self.external_embedding) + 3, self.edim))
			for word, i in self.extrnd.iteritems():
				self.model["extrn-lookup"].init_row(i, self.external_embedding[word])
			self.extrnd['*PAD*'] = 1
			self.extrnd['*INITIAL*'] = 2

			print('Load external embedding. Vector dimensions', self.edim)

		self.vocab['*PAD*'] = 1
		self.pos[0]['*PAD*'] = 1
		self.pos[1]['*PAD*'] = 1

		self.vocab['*INITIAL*'] = 2
		self.pos[0]['*INITIAL*'] = 2
		self.pos[1]['*INITIAL*'] = 2

		if self.embed_mode%2 == 0:
			self.wlookup=[self.model.add_lookup_parameters((len(words) + 3, self.wdims),name="word-lookup0"),
					  self.model.add_lookup_parameters((len(words) + 3, self.wdims), name="word-lookup1")]
		else:
			self.wlookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims), name="word-lookup")
		self.plookup=[self.model.add_lookup_parameters((len(pos1) + 3, self.pdims),name="pos-lookup0"),
					  self.model.add_lookup_parameters((len(pos2) + 3, self.pdims),name="pos-lookup1")]
		if self.rep_mode==2:
			self.rlookup = [self.model.add_lookup_parameters((len(rels1)+2, self.rdims), name="rels-lookup1"),
							self.model.add_lookup_parameters((len(rels2)+2, self.rdims), name="rels-lookup2")]
		elif self.rep_mode<2:
			self.rlookup = [self.model.add_lookup_parameters((len(rels1), self.rdims), name="rels-lookup1"),
							self.model.add_lookup_parameters((len(rels2), self.rdims), name="rels-lookup2")]

		self.builders = [[LSTMBuilder(self.layers, self.ldims, self.ldims, self.model),
						  LSTMBuilder(self.layers, self.ldims, self.ldims, self.model)],
						 [LSTMBuilder(self.layers, self.ldims, self.ldims, self.model),
						  LSTMBuilder(self.layers, self.ldims, self.ldims, self.model)]]

		self.blstmFlag = options.blstmFlag
		if self.blstmFlag:
			if self.blstm_mode>0:
				self.commonLSTM = [LSTMBuilder(self.blstm_mode, self.ldims, self.ldims * 0.5, self.model),
								   LSTMBuilder(self.blstm_mode, self.ldims, self.ldims * 0.5, self.model)]

			if self.blstm_mode<2:
				self.surfaceBuilders = [[LSTMBuilder(self.layers - self.blstm_mode, self.ldims, self.ldims * 0.5, self.model),
										 LSTMBuilder(self.layers - self.blstm_mode, self.ldims, self.ldims * 0.5, self.model)],
										[LSTMBuilder(self.layers - self.blstm_mode, self.ldims, self.ldims * 0.5, self.model),
										 LSTMBuilder(self.layers - self.blstm_mode, self.ldims, self.ldims * 0.5, self.model)]]

		self.nnvecs = 2
		if self.embed_mode // 2 == 0:
			self.word2lstm=[self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)),name="word-to-lstm1"),
						self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)), name="word-to-lstm2")]
		else:
			self.word2lstm = [self.model.add_parameters(
				(self.ldims, self.wdims + 2*self.pdims + (self.edim if self.external_embedding is not None else 0)),
				name="word-to-lstm1"),
							  self.model.add_parameters((self.ldims, self.wdims + 2*self.pdims + (
								  self.edim if self.external_embedding is not None else 0)), name="word-to-lstm2")]
		self.word2lstmbias = [self.model.add_parameters((self.ldims),name="word-to-lstm-bias1"),
							  self.model.add_parameters((self.ldims), name="word-to-lstm-bias2")]
		self.lstm2lstm=[self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims),name="lstm-to-lstm1"),
						self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims),name="lstm-to-lstm2")]
		self.lstm2lstmbias = [self.model.add_parameters((self.ldims),name="lstm-to-lstm-bias1"),
							  self.model.add_parameters((self.ldims),name="lstm-to-lstm-bias2")]

		if self.ul_mode==0:
			if self.rep_mode==0:
				self.hidLayer = [self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="hidden-layer0"),
								self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="hidden-layer1")]
			elif self.rep_mode==1:
				self.hidLayer = [self.model.add_parameters((self.hidden_units, 2 * self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="hidden-layer0"),
								self.model.add_parameters((self.hidden_units, 2 * self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="hidden-layer1")]
			elif self.rep_mode==2:
				self.hidLayer = [self.model.add_parameters((self.hidden_units, (2 * self.ldims + self.rdims) * self.nnvecs * ((self.k + 1) * 2)),name="hidden-layer0"),
					self.model.add_parameters((self.hidden_units, (2 * self.ldims + self.rdims) * self.nnvecs * ((self.k + 1) * 2)),name="hidden-layer1")]

			self.hidBias = [self.model.add_parameters((self.hidden_units), name="hidden-bias0"),
							self.model.add_parameters((self.hidden_units), name="hidden-bias1")]

			self.hid2Layer = [self.model.add_parameters((self.hidden2_units, self.hidden_units), name="hidden2-layer0"),
							  self.model.add_parameters((self.hidden2_units, self.hidden_units), name="hidden2-layer1")]
			self.hid2Bias = [self.model.add_parameters((self.hidden2_units), name="hidden2-bias0"),
							 self.model.add_parameters((self.hidden2_units), name="hidden2-bias1")]

			self.outLayer = [self.model.add_parameters((2, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units),name="output-layer0"),
							self.model.add_parameters((2, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units),name="output-layer1")]
			self.outBias = [self.model.add_parameters((2), name="output-bias0"),
							self.model.add_parameters((2), name="output-bias1")]
		elif self.ul_mode==1:
			if self.rep_mode == 0:
				self.hidLayer = self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)), name="hidden-layer")
			elif self.rep_mode == 1:
				self.hidLayer = self.model.add_parameters((self.hidden_units, 2 * self.ldims * self.nnvecs * ((self.k + 1) * 2)), name="hidden-layer")

			elif self.rep_mode==2:
				self.hidLayer = self.model.add_parameters((self.hidden_units, 2 * (self.nnvecs * self.ldims + self.rdims) * ((self.k + 1) * 2)), name="hidden-layer")

			self.hidBias = self.model.add_parameters((self.hidden_units), name="hidden-bias")

			self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units), name="hidden2-layer")
			self.hid2Bias = self.model.add_parameters((self.hidden2_units), name="hidden2-bias")

			self.outLayer = self.model.add_parameters((2, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units), name="output-layer")
			self.outBias = self.model.add_parameters((2), name="output-bias")

		if self.rep_mode==0:
			self.rhidLayer = [self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="rhidden-layer1"),
							self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="rhidden-layer2")]
		elif self.rep_mode==1:
			self.rhidLayer = [self.model.add_parameters((self.hidden_units, 2 * self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="rhidden-layer1"),
							self.model.add_parameters((self.hidden_units, 2 * self.ldims * self.nnvecs * ((self.k + 1) * 2)),name="rhidden-layer2")]
		elif self.rep_mode==2:
			self.rhidLayer = [self.model.add_parameters((self.hidden_units, 2 * (2 * self.ldims + self.rdims) * ((self.k + 1) * 2)),name="rhidden-layer1"),
							self.model.add_parameters((self.hidden_units, 2 * (2 * self.ldims + self.rdims) * ((self.k + 1) * 2)),name="rhidden-layer2")]

		self.rhidBias=[self.model.add_parameters((self.hidden_units),name="rhidden-bias1"),
					   self.model.add_parameters((self.hidden_units),name="rhidden-bias2")]

		self.rhid2Layer=[self.model.add_parameters( (self.hidden2_units, self.hidden_units),name="rhidden2-layer1"),
						 self.model.add_parameters( (self.hidden2_units, self.hidden_units),name="rhidden2-layer2")]
		self.rhid2Bias=[self.model.add_parameters( (self.hidden2_units),name="rhidden2-bias1"),
						self.model.add_parameters( (self.hidden2_units),name="rhidden2-bias2")]

		self.routLayer=[self.model.add_parameters( (2 * (len(self.irels[0]) + 0), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units),name="routput-layer1"),
						self.model.add_parameters( (2 * (len(self.irels[1]) + 0), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units),name="routput-layer2")]
		self.routBias=[self.model.add_parameters( (2 * (len(self.irels[0]) + 0)),name="routput-bias1"),
					   self.model.add_parameters( (2 * (len(self.irels[1]) + 0)),name="routput-bias2")]


	def  __getExpr(self, forest, support_forest, i, train, tree_dim,indexing_lists):
		roots = [forest.roots[idx] for idx in indexing_lists[tree_dim]]
		support_roots = [support_forest.roots[idx] for idx in indexing_lists[tree_dim]]
		nRoots = len(roots)

		if self.builders is None:
			input = concatenate([ concatenate(roots[j].lstms) if j>=0 and j<nRoots else self.empty[tree_dim] for j in range(i-self.k, i+self.k+2) ])
		else:
			if self.rep_mode==0:
				input = concatenate([concatenate([roots[j].lstms[0].output(), roots[j].lstms[1].output()])
						if j >= 0 and j < nRoots else self.empty[tree_dim] for j in range(i - self.k, i + self.k + 2)])
			elif self.rep_mode==1:
				filler = concatenate([self.empty[tree_dim], self.empty[1 - tree_dim]])
				input = concatenate([concatenate([roots[j].lstms[0].output(), roots[j].lstms[1].output(), support_roots[j].lstms[0].output(),
								  support_roots[j].lstms[1].output()]) if j >= 0 and j < nRoots else filler for j in range(i - self.k, i + self.k + 2)])
			elif self.rep_mode==2:
				print(nRoots,i,self.k)
				for lol in range(i-self.k,i+self.k+2):
					if 0<=lol<nRoots:
						print(roots[lol].pred_irel,support_roots[lol].pred_irel)

				filler = concatenate([self.empty[tree_dim], lookup(self.rlookup[tree_dim], len(self.irels[tree_dim]) + 1),
						self.empty[1 - tree_dim],
						lookup(self.rlookup[1 - tree_dim], len(self.irels[1 - tree_dim]) + 1)])
				input = concatenate([concatenate([roots[j].lstms[0].output(), roots[j].lstms[1].output(),lookup(self.rlookup[tree_dim],
						len(self.irels[tree_dim])) if roots[j].pred_irel == None else lookup(self.rlookup[tree_dim], roots[j].pred_irel),
						support_roots[j].lstms[0].output(), support_roots[j].lstms[1].output(),
						lookup(self.rlookup[1 - tree_dim], len(self.irels[1 - tree_dim])) if
						support_roots[j].pred_irel == None else lookup(self.rlookup[1 - tree_dim],support_roots[j].pred_irel)])
						if j >= 0 and j < nRoots else filler for j in range(i - self.k, i + self.k + 2)])

		if self.hidden2_units > 0:
			routput = (self.routLayer[tree_dim] * self.activation(self.rhid2Bias[tree_dim] + self.rhid2Layer[tree_dim] * self.activation(self.rhidLayer[tree_dim] * input + self.rhidBias[tree_dim])) + self.routBias[tree_dim])
		else:
			routput = (self.routLayer[tree_dim] * self.activation(self.rhidLayer[tree_dim] * input + self.rhidBias[tree_dim]) + self.routBias[tree_dim])

		if self.ul_mode==0:
			if self.hidden2_units > 0:
				output = (self.outLayer[tree_dim] * self.activation(self.hid2Bias[tree_dim] + self.hid2Layer[tree_dim] * self.activation(self.hidLayer[tree_dim] * input + self.hidBias[tree_dim])) + self.outBias[tree_dim])
			else:
				output = (self.outLayer[tree_dim] * self.activation(self.hidLayer[tree_dim] * input + self.hidBias[tree_dim]) + self.outBias[tree_dim])

		elif self.ul_mode==1:
			if self.hidden2_units > 0:
				output = (self.outLayer * self.activation(self.hid2Bias + self.hid2Layer * self.activation(self.hidLayer * input + self.hidBias)) + self.outBias)
			else:
				output = (self.outLayer * self.activation(self.hidLayer * input + self.hidBias) + self.outBias)

		return routput, output


	def __evaluate(self, forest, support_forest, train,tree_dim,indexing_lists):
		nRoots = len(indexing_lists[tree_dim])
		nRels = len(self.irels[tree_dim])
		for i in range(nRoots - 1):
			if forest.roots[indexing_lists[tree_dim][i]].scores is None:
				output, uoutput = self.__getExpr(forest,support_forest, i, train,tree_dim,indexing_lists)
				scrs = output.value()
				uscrs = uoutput.value()
				forest.roots[indexing_lists[tree_dim][i]].exprs = [(pick(output, j * 2) + pick(uoutput, 0), pick(output, j * 2 + 1) + pick(uoutput, 1)) for j in range(len(self.irels[tree_dim]))]
				forest.roots[indexing_lists[tree_dim][i]].scores = [(scrs[j * 2] + uscrs[0], scrs[j * 2 + 1] + uscrs[1]) for j in range(len(self.irels[tree_dim]))]

	def getParams(self):
		p=[]
		for i in self.model.parameters_list():
			p.append(i.as_array())
			print(i.name())
		pickle.dump(p,open('p.pkl','wb'))
		exit()


	def Save(self, filename):
		self.model.save(filename)


	def Load(self, filename):
		self.model.populate(filename)


	def Init(self):
		evec = self.elookup[1] if self.external_embedding is not None else None

		paddingPosVec_0 = self.plookup[0][1] if self.pdims > 0 else None
		paddingPosVec_1 = self.plookup[1][1] if self.pdims > 0 else None
		if self.embed_mode //2 == 0 :
			if self.embed_mode==0:
				paddingWordVec_0 = self.wlookup[0][1]
				paddingWordVec_1 = self.wlookup[1][1]
				paddingVec_0 = tanh(self.word2lstm[0] * concatenate(list(filter(None, [paddingWordVec_0, paddingPosVec_0, evec]))) + self.word2lstmbias[0] )
				paddingVec_1 = tanh(self.word2lstm[1] * concatenate(list(filter(None, [paddingWordVec_1, paddingPosVec_1, evec]))) + self.word2lstmbias[1] )
			elif self.embed_mode==1:
				paddingWordVec = self.wlookup[1]
				paddingVec_0 = tanh(self.word2lstm[0] * concatenate(list(filter(None, [paddingWordVec, paddingPosVec_0, evec]))) + self.word2lstmbias[0])
				paddingVec_1 = tanh(self.word2lstm[1] * concatenate(list(filter(None, [paddingWordVec, paddingPosVec_1, evec]))) + self.word2lstmbias[1])
		else:
			if self.embed_mode==3:
				paddingWordVec_0 = self.wlookup[0][1]
				paddingWordVec_1 = self.wlookup[1][1]
				paddingVec_0 = tanh(self.word2lstm[0] * concatenate(list(filter(None, [paddingWordVec_0, paddingPosVec_0,paddingPosVec_1, evec]))) + self.word2lstmbias[0] )
				paddingVec_1 = tanh(self.word2lstm[1] * concatenate(list(filter(None, [paddingWordVec_1, paddingPosVec_1,paddingPosVec_0, evec]))) + self.word2lstmbias[1] )
			elif self.embed_mode==4:
				paddingWordVec = self.wlookup[1]
				paddingVec_0 = tanh(self.word2lstm[0] * concatenate(list(filter(None, [paddingWordVec, paddingPosVec_0,paddingPosVec_1, evec]))) + self.word2lstmbias[0])
				paddingVec_1 = tanh(self.word2lstm[1] * concatenate(list(filter(None, [paddingWordVec, paddingPosVec_1,paddingPosVec_0, evec]))) + self.word2lstmbias[1])

		self.empty = [(concatenate([self.builders[0][0].initial_state().add_input(paddingVec_0).output(), self.builders[0][1].initial_state().add_input(paddingVec_0).output()])),
				  (concatenate([self.builders[1][0].initial_state().add_input(paddingVec_1).output(), self.builders[1][1].initial_state().add_input(paddingVec_1).output()]))]


	def getWordEmbeddings(self, forest, support_forest, train,tree_dim):
		for root,support_root in zip(forest.roots,support_forest.roots):
			c = float(self.wordsCount.get(root.norm, 0))
			dropFlag =  not train or (random.random() < (c/(0.25+c)))
			if self.embed_mode%2 == 0:
				root.wordvec = self.wlookup[tree_dim][int(self.vocab.get(root.norm, 0)) if dropFlag else 0]
			elif self.embed_mode%2 == 1:
				root.wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if dropFlag else 0]

			root.posvec = self.plookup[tree_dim][int(self.pos[tree_dim][root.pos])] if self.pdims > 0 else None
			if self.embed_mode//2 == 1:
				support_root.posvec=self.plookup[1-tree_dim][int(self.pos[1-tree_dim][support_root.pos])] if self.pdims>0 else None

			root.evec = None # external embedding
			if self.embed_mode//2==0:
				root.ivec = (self.word2lstm[tree_dim] * concatenate(list(filter(None, [root.wordvec, root.posvec, root.evec])))) + self.word2lstmbias[tree_dim]
			else:
				root.ivec = (self.word2lstm[tree_dim] * concatenate(list(filter(None, [root.wordvec, root.posvec,support_root.posvec, root.evec])))) + self.word2lstmbias[tree_dim]
		if self.blstmFlag:
			if self.blstm_mode==0:
				forward  = self.surfaceBuilders[tree_dim][0].initial_state()
				backward = self.surfaceBuilders[tree_dim][1].initial_state()

				for froot, rroot in zip(forest.roots, reversed(forest.roots)):
					forward = forward.add_input( froot.ivec )
					backward = backward.add_input( rroot.ivec )
					froot.fvec = forward.output()
					rroot.bvec = backward.output()
				for root in forest.roots:
					root.vec = concatenate( [root.fvec, root.bvec] )

			elif self.blstm_mode==1:
				forward = self.commonLSTM[0].initial_state()
				backward = self.commonLSTM[1].initial_state()

				for froot, rroot in zip(forest.roots, reversed(forest.roots)):
					forward = forward.add_input(froot.ivec)
					backward = backward.add_input(rroot.ivec)
					froot.fvec_common = forward.output()
					rroot.bvec_common = backward.output()
				for root in forest.roots:
					root.vec_common = concatenate([root.fvec_common, root.bvec_common])

				forward = self.surfaceBuilders[tree_dim][0].initial_state()
				backward = self.surfaceBuilders[tree_dim][1].initial_state()

				for froot, rroot in zip(forest.roots, reversed(forest.roots)):
					forward = forward.add_input(froot.vec_common)
					backward = backward.add_input(rroot.vec_common)
					froot.fvec = forward.output()
					rroot.bvec = backward.output()
				for root in forest.roots:
					root.vec = concatenate([root.fvec, root.bvec])

			elif self.blstm_mode==2:
				forward = self.commonLSTM[0].initial_state()
				backward = self.commonLSTM[1].initial_state()

				for froot, rroot in zip(forest.roots, reversed(forest.roots)):
					forward = forward.add_input(froot.ivec)
					backward = backward.add_input(rroot.ivec)
					froot.fvec_common = forward.output()
					rroot.bvec_common = backward.output()
				for root in forest.roots:
					root.vec = concatenate([root.fvec_common, root.bvec_common])

		else:
			for root in forest.roots:
				root.vec = tanh( root.ivec )

	def __pred_best_score(self, forest, tree_dim, indexing_list):
		if len(indexing_list) <= 1:
			return [float("-inf"), None, None, None, None, None, None, None]
		bestValidOp, bestValidScore = None, float("-inf")
		bestWrongOp, bestWrongScore = None, float("-inf")
		bestValidExpr, bestWrongExpr = None, None

		bestValidParent, bestValidChild = None, None
		bestWrongParent, bestWrongChild = None, None
		bestValidIndex, bestWrongIndex = None, None

		bestWrongIRel, bestWrongRel = None, None
		bestValidIRel, bestValidRel = None, None
		roots = [forest.roots[idx] for idx in indexing_list]

		rootsIds = set([root.id for root in roots])

		for i in range(len(indexing_list) - 1):
			for irel, rel in enumerate(self.irels[tree_dim]):
				for op in range(2):
					child = i + (1 - op)
					parent = i + op

					if True:
						if bestValidScore < roots[i].scores[irel][op]:
							bestValidScore = roots[i].scores[irel][op]
							bestValidOp = op
							bestValidParent, bestValidChild = parent, child
							bestValidIndex = i
							bestValidIRel, bestValidRel = irel, rel
							bestValidExpr = roots[bestValidIndex].exprs[bestValidIRel][bestValidOp]

		# Parent, child and index are the indices w.r.t the indexing_list, i.e roots[indexing_list[parent/child/index]] points to the correct parent/child/index in the retained forest.
		bestValidpack = [bestValidScore, bestValidExpr, bestValidOp, bestValidParent, bestValidChild, bestValidIndex,
						 bestValidIRel, bestValidRel]
		bestWrongpack = [bestWrongScore, bestWrongExpr, bestWrongOp, bestWrongParent, bestWrongChild, bestWrongIndex,
						 bestWrongIRel, bestWrongRel]
		return bestValidpack

	def  __pick_best_scores(self,forest,unassigned,tree_dim,indexing_list):
		if len(indexing_list)<=1:
			return [[float("-inf"),None,None,None,None,None,None,None],[float("-inf"),None,None,None,None,None,None,None]]
		bestValidOp, bestValidScore = None, float("-inf")
		bestWrongOp, bestWrongScore = None, float("-inf")
		bestValidExpr, bestWrongExpr = None, None

		bestValidParent, bestValidChild = None, None
		bestWrongParent, bestWrongChild = None, None
		bestValidIndex, bestWrongIndex = None, None

		bestWrongIRel, bestWrongRel = None,None
		bestValidIRel, bestValidRel = None,None
		roots = [forest.roots[idx] for idx in indexing_list]

		rootsIds = set([root.id for root in roots])

		for i in range(len(indexing_list) - 1):
			for irel, rel in enumerate(self.irels[tree_dim]):
				for op in range(2):
					child = i + (1 - op)
					parent = i + op

					oracleCost = unassigned[roots[child].id] + (
						0 if roots[child].parent_id not in rootsIds or roots[child].parent_id == roots[
							parent].id else 1)

					# print (i,irel,op)
					# print(roots[i].scores[irel][op])
					if oracleCost == 0 and (roots[child].parent_id != roots[parent].id or roots[child].relation == rel):
						if bestValidScore < roots[i].scores[irel][op]:
							bestValidScore = roots[i].scores[irel][op]
							bestValidOp = op
							bestValidParent, bestValidChild = parent, child
							bestValidIndex = i
							bestValidIRel, bestValidRel = irel, rel
							bestValidExpr = roots[bestValidIndex].exprs[bestValidIRel][bestValidOp]

					elif bestWrongScore < roots[i].scores[irel][op]:
						bestWrongScore = roots[i].scores[irel][op]
						bestWrongParent, bestWrongChild = parent, child
						bestWrongOp = op
						bestWrongIndex = i
						bestWrongIRel, bestWrongRel = irel, rel
						bestWrongExpr = roots[bestWrongIndex].exprs[bestWrongIRel][bestWrongOp]
		# Parent, child and index are the indices w.r.t the indexing_list, i.e roots[indexing_list[parent/child/index]] points to the correct parent/child/index in the retained forest.
		bestValidpack=[bestValidScore,bestValidExpr,bestValidOp,bestValidParent,bestValidChild,bestValidIndex,bestValidIRel,bestValidRel]
		bestWrongpack = [bestWrongScore, bestWrongExpr, bestWrongOp, bestWrongParent, bestWrongChild, bestWrongIndex,bestWrongIRel, bestWrongRel]
		return [bestValidpack,bestWrongpack]


	def Predict(self, conll_path1, conll_path2):
		conllFP1 = open(conll_path1, 'r')
		conllFP2 = open(conll_path2, 'r')
		data = list(zip(list(read_conll(conllFP1, False)), list(read_conll(conllFP2, False))))

		for iSentence, sentences in enumerate(data):
			self.Init()
			selected_dims = []
			sentence_0 = sentences[0]
			sentence_1 = sentences[1]
			forest0 = ParseForest(sentence_0)
			forest1 = ParseForest(sentence_1)
			forests = [forest0, forest1]

			self.getWordEmbeddings(forests[0],forests[1], False, 0)
			self.getWordEmbeddings(forests[1],forests[0], False, 1)

			for c1 in range(2):
				for root in forests[c1].roots:
					root.lstms = [self.builders[c1][0].initial_state().add_input(root.vec),
								  self.builders[c1][1].initial_state().add_input(root.vec)]

			indexing_lists = [[i for i in range(len(forests[0].roots))], [i for i in range(len(forests[1].roots))]]

			while (len(indexing_lists[0]) > 1) or (len(indexing_lists[1]) > 1):
				self.__evaluate(forests[0], forests[1], False, 0, indexing_lists)
				self.__evaluate(forests[1], forests[0], False, 1, indexing_lists)

				score_pack = [[], []]
				score_pack[0] = self.__pred_best_score(forests[0], 0, indexing_lists[0])
				score_pack[1] = self.__pred_best_score(forests[1], 1, indexing_lists[1])

				selectedDim = 0 if score_pack[0][0] > score_pack[1][0] else 1
				selectedOp = score_pack[selectedDim][2]
				selectedParent = score_pack[selectedDim][3]
				selectedChild = score_pack[selectedDim][4]
				selectedIndex = score_pack[selectedDim][5]
				selectedIRel, selectedRel = score_pack[selectedDim][6], score_pack[selectedDim][7]

				roots = [forests[selectedDim].roots[idx] for idx in indexing_lists[selectedDim]]

				for j in range(max(0, selectedIndex - self.k - 1),
								min(len(indexing_lists[selectedDim]), selectedIndex + self.k + 2)):
					roots[j].scores = None

				roots[selectedChild].pred_parent_id = roots[selectedParent].id
				roots[selectedChild].pred_relation = selectedRel
				if (selectedRel == None):
					print(score_pack[selectedDim][0])
				roots[selectedParent].lstms[selectedOp] = roots[selectedParent].lstms[selectedOp].add_input(
					self.activation(self.lstm2lstm[selectedDim] *
									noise(concatenate(
										[roots[selectedChild].lstms[0].output(),
										 lookup(self.rlookup[selectedDim], selectedIRel),
										 roots[selectedChild].lstms[1].output()]), 0.0) + self.lstm2lstmbias[
										selectedDim]))

				selected_dims.append(selectedDim)
				forests[selectedDim].Attach(indexing_lists[selectedDim][selectedParent],
											indexing_lists[selectedDim][selectedChild], selectedRel)
				del indexing_lists[selectedDim][selectedChild]
			if iSentence % 100 == 0:
				print
				"Number of sentences processed : " + str(iSentence + 1)
			renew_cg()
			yield (sentences, selected_dims)

	def Train(self, conll_path1, conll_path2):
		mloss = 0.0
		errors = 0
		batch = 0
		eloss = 0.0
		eerrors = 0
		lerrors = 0
		etotal = 0
		ltotal = 0

		start = time.time()

		# TRAIN ROUTINE
		conllFP1 = open(conll_path1, 'r')
		conllFP2 = open(conll_path2, 'r')
		shuffledData = list(zip(list(read_conll(conllFP1, False)), list(read_conll(conllFP2, False))))
		random.shuffle(shuffledData)

		errs = []
		eeloss = 0.0

		self.Init()
		for iSentence, sentences in enumerate(shuffledData):
			if iSentence % 100 == 0 and iSentence != 0:
				print("model_"+str(self.embed_mode)+str(self.blstm_mode)+str(self.rep_mode)+str(self.ul_mode)+"  "+'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (
					float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal), 'Time', time.time() - start)
				start = time.time()
				eerrors = 0
				eloss = 0.0
				etotal = 0
				lerrors = 0
				ltotal = 0

			sentence_0 = sentences[0]
			sentence_1 = sentences[1]
			forest0 = ParseForest(sentence_0)
			forest1 = ParseForest(sentence_1)
			forests = [forest0, forest1]

			self.getWordEmbeddings(forests[0],forests[1], True, 0)
			self.getWordEmbeddings(forests[1],forests[0], True, 1)

			for c1 in range(2):
				for root in forests[c1].roots:
					root.lstms = [self.builders[c1][0].initial_state().add_input(root.vec),
								  self.builders[c1][1].initial_state().add_input(root.vec)]

			unassigned = [
				{entry.id: sum([1 for pentry in sentence_0 if pentry.parent_id == entry.id]) for entry in sentence_0},
				{entry.id: sum([1 for pentry in sentence_1 if pentry.parent_id == entry.id]) for entry in sentence_1}]

			indexing_lists = [[i for i in range(len(forests[0].roots))], [i for i in range(len(forests[1].roots))]]

			if self.parse_mode==0:
				parity_bit=self.priority_dim

			while self.parse_mode == 2 and len(indexing_lists[self.priority_dim])>1:
				selectedDim = self.priority_dim
				found_flag = False
				for c1 in range(len(indexing_lists[self.priority_dim]) - 1):
					for c2 in range(c1+1,len(indexing_lists[self.priority_dim]) - 1):
						if sentences[self.priority_dim][indexing_lists[c1]].parent_id == sentences[self.priority_dim][
							indexing_lists[c2]].id and unassigned[selectedDim][sentences[self.priority_dim][indexing_lists[c1]].id]==0:
							selectedOp = 1
							selectedParent = c2
							selectedChild = c1
							selectedIRel, selectedRel = self.irels[self.priority_dim][
															sentences[self.priority_dim][indexing_lists[c1]].relation], \
														sentences[self.priority_dim][indexing_lists[c1]].relation
							found_flag = True
							break
						elif sentences[self.priority_dim][indexing_lists[c2]].parent_id == sentences[self.priority_dim][indexing_lists[c1]].id:
							selectedOp = 0
							selectedParent = c1
							selectedChild = c2
							selectedIRel, selectedRel = self.irels[self.priority_dim][sentences[self.priority_dim][indexing_lists[c2]].relation],sentences[self.priority_dim][indexing_lists[c2]].relation
							found_flag=True
							break
				if found_flag==False:
					print("didn't find a tree to make, incorrect CoNLL entry")

				roots = [forests[selectedDim].roots[idx] for idx in indexing_lists[selectedDim]]

				unassigned[selectedDim][roots[selectedChild].parent_id] -= 1

				roots[selectedParent].lstms[selectedOp] = roots[selectedParent].lstms[selectedOp].add_input(
					self.activation(self.lstm2lstm[selectedDim] *
									noise(concatenate(
										[roots[selectedChild].lstms[0].output(),
										 lookup(self.rlookup[selectedDim], selectedIRel),
										 roots[selectedChild].lstms[1].output()]), 0.0) + self.lstm2lstmbias[
										selectedDim]))

				# rectify the update with indexing_lists
				forests[selectedDim].Attach(indexing_lists[selectedDim][selectedParent],
											indexing_lists[selectedDim][selectedChild], selectedIRel)
				# delete the entry from the corresponding indexing_list
				del indexing_lists[selectedDim][selectedChild]

			while (len(indexing_lists[0]) > 1) or (len(indexing_lists[1]) > 1):
				self.__evaluate(forests[0], forests[1], True, 0, indexing_lists)
				self.__evaluate(forests[1], forests[0], True, 1, indexing_lists)

				score_pack = [[], []]
				score_pack[0] = self.__pick_best_scores(forests[0], unassigned[0], 0, indexing_lists[0])
				score_pack[1] = self.__pick_best_scores(forests[1], unassigned[1], 1, indexing_lists[1])
				# __pick_best_scores returns a list of two lists, each containing [Score,Expr,Op,Parent,Child,Index,IRel,Rel]

				bestValidDim = 0 if score_pack[0][0][0] > score_pack[1][0][0] else 1
				bestWrongDim = 0 if score_pack[0][1][0] > score_pack[1][1][0] else 1

				if self.parse_mode==0:
					bestValidDim = parity_bit
					bestWrongDim = parity_bit
					parity_bit = 1 - parity_bit

				elif self.parse_mode==1:
					bestValidDim = self.priority_dim if len(indexing_lists[self.priority_dim]) > 1 else 1 - self.priority_dim
					bestWrongDim = self.priority_dim if len(indexing_lists[self.priority_dim]) > 1 else 1 - self.priority_dim

				if score_pack[bestValidDim][0][0] < score_pack[bestWrongDim][1][0] + 1.0:
					loss = score_pack[bestWrongDim][1][1] - score_pack[bestValidDim][0][1]
					mloss += 1.0 + score_pack[bestWrongDim][1][0] - score_pack[bestValidDim][0][0]
					eloss += 1.0 + score_pack[bestWrongDim][1][0] - score_pack[bestValidDim][0][0]
					errs.append(loss)

				if not self.oracle or score_pack[bestValidDim][0][0] - score_pack[bestWrongDim][1][0] > 1.0 or (
						score_pack[bestValidDim][0][0] > score_pack[bestWrongDim][1][0] and random.random() > 0.1):
					selectedDim = bestValidDim
					selectedOp = score_pack[bestValidDim][0][2]
					selectedParent = score_pack[bestValidDim][0][3]
					selectedChild = score_pack[bestValidDim][0][4]
					selectedIndex = score_pack[bestValidDim][0][5]
					selectedIRel, selectedRel = score_pack[bestValidDim][0][6], score_pack[bestValidDim][0][7]
				else:
					selectedDim = bestWrongDim
					selectedOp = score_pack[bestWrongDim][1][2]
					selectedParent = score_pack[bestWrongDim][1][3]
					selectedChild = score_pack[bestWrongDim][1][4]
					selectedIndex = score_pack[bestWrongDim][1][5]
					selectedIRel, selectedRel = score_pack[bestWrongDim][1][6], score_pack[bestWrongDim][1][7]

				roots = [forests[selectedDim].roots[idx] for idx in indexing_lists[selectedDim]]
				if roots[selectedChild].parent_id != roots[selectedParent].id or selectedRel != roots[
					selectedChild].relation:
					lerrors += 1
					if roots[selectedChild].parent_id != roots[selectedParent].id:
						errors += 1
						eerrors += 1

				etotal += 1

				for j in range(max(0, selectedIndex - self.k - 1),
								min(len(indexing_lists[selectedDim]), selectedIndex + self.k + 2)):
					roots[j].scores = None

				unassigned[selectedDim][roots[selectedChild].parent_id] -= 1

				roots[selectedParent].lstms[selectedOp] = roots[selectedParent].lstms[selectedOp].add_input(
					self.activation(self.lstm2lstm[selectedDim] *
									noise(concatenate(
										[roots[selectedChild].lstms[0].output(),
										 lookup(self.rlookup[selectedDim], selectedIRel),
										 roots[selectedChild].lstms[1].output()]), 0.0) + self.lstm2lstmbias[selectedDim]))

				# rectify the update with indexing_lists
				forests[selectedDim].Attach(indexing_lists[selectedDim][selectedParent],
											indexing_lists[selectedDim][selectedChild],selectedIRel)
				# delete the entry from the corresponding indexing_list
				del indexing_lists[selectedDim][selectedChild]

			if len(errs) > 50.0:
				eerrs = ((esum(errs)) * (1.0 / (float(len(errs)))))
				scalar_loss = eerrs.scalar_value()
				eerrs.backward()
				self.trainer.update()
				errs = []
				lerrs = []

				renew_cg()
				self.Init()

		if len(errs) > 0:
			eerrs = (esum(errs)) * (1.0 / (float(len(errs))))
			eerrs.scalar_value()
			eerrs.backward()
			self.trainer.update()

			errs = []
			lerrs = []

			renew_cg()

		# self.trainer.update_epoch()
		print("Loss: ", mloss / iSentence)
