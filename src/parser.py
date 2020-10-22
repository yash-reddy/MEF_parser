from optparse import OptionParser
import json, utils, easylstm, os, pickle, time

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train1", dest="conll_train1", help="Annotated CONLL train file", metavar="FILE", default="data/catib/catib.train.v2.11062020.proj.conllu")
    parser.add_option("--train2", dest="conll_train2", help="Annotated CONLL train file", metavar="FILE",default="data/ud/ud.train.v2.11062020.proj.conllu")
    parser.add_option("--dev1", dest="conll_dev1", help="Annotated CONLL dev file", metavar="FILE", default="data/catib/catib.dev.v2.11062020.proj.conllu")
    parser.add_option("--dev2", dest="conll_dev2", help="Annotated CONLL dev file", metavar="FILE", default="data/ud/ud.dev.v2.11062020.proj.conllu")
    parser.add_option("--test1", dest="conll_test1", help="Annotated CONLL test file", metavar="FILE", default="data/PTB_SD_3_3_0/test.conll")
    parser.add_option("--test2", dest="conll_test2", help="Annotated CONLL test file", metavar="FILE", default="data/PTB_SD_3_3_0/test.conll")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="easyfirst.model")
    parser.add_option("--loadmodel", dest="model", help="Load/Save model file", metavar="FILE", default="results/easyfirst.model")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=40)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--k", type="int", dest="window", default=1)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=200)
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--cnn-seed", type="int", dest="seed", default=0)
    parser.add_option("--devgen", action="store_true", dest="devgenFlag", default=False)
    parser.add_option("--startepoch", type="int", dest="start_epoch", default=0)
    parser.add_option("--embedmode", type="int", dest="embed_mode", default=0) # 0 for disjoint embeddings, 1 for common word embeddings, 2 for (sharing POS tags, disjoint word embeddings), 3 for (shared POS tags, common word embeddings)
    parser.add_option("--blstmmode", type="int", dest="blstm_mode", default=0)
    parser.add_option("--repmode", type="int", dest="rep_mode", default=0)
    parser.add_option("--ulmode", type="int", dest="ul_mode", default=0) # 0 for disjoint, 1 for unlabeled scoring shared
    parser.add_option("--parsemode", type="int", dest="parse_mode", default=-1) # 0 for normal, 1 for absolute parity, 2 for pipeline parsing, 3 for single dimension parsing with the other dimension as feature inputs
    parser.add_option("--priority", type="int", dest="priority_dim", default=-1)

    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding

    if options.devgenFlag:
        print 'Preparing vocab'
        words, w2i, pos1, rels1 = utils.vocab(options.conll_train1)
        _1, _2, pos2, rels2 = utils.vocab(options.conll_train2)
        for epoch in xrange(options.start_epoch,options.epochs):
            model_params = os.path.join(options.output, "easyfirst.model" + str(epoch + 1))
            if not os.path.exists(model_params):
                print(model_params + " doesn't exist, exiting process")
                break
            print "Generating dev predictions for epoch : "+str(epoch+1)
            print(options.params)
            params=os.path.join(options.output,options.params)
            #with open(params, 'r') as paramsfp:
            #    words, w2i, pos1, pos2, rels1, rels2, stored_opt = pickle.load(paramsfp)

            print 'Initializing Hierarchical Tree LSTM parser:'
            parser = easylstm.EasyFirstLSTM(words, pos1, pos2, rels1, rels2, w2i, options)
            parser.Load(model_params)
            dev_path1,dev_path2= options.conll_dev1,options.conll_dev2
            write_path1,write_path2=os.path.join(options.output,'dev_0_epoch_' + str(epoch+1) + '.conll'),os.path.join(options.output,'dev_1_epoch_' + str(epoch+1) + '.conll')
            dim_tracker_path=os.path.join(options.output,"dim_tracker_epoch_"+str(epoch+1)+".pkl")
            utils.get_results(parser,dev_path1,dev_path2,write_path1,write_path2,dim_tracker_path)

    elif options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos1, pos2, rels1, rels2, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        print 'Initializing Hierarchical Tree LSTM parser:'
        parser = easylstm.EasyFirstLSTM(words, pos1, pos2, rels1,rels2, w2i, stored_opt)

        parser.Load(options.loadmodel)
        tespath = os.path.join(options.output, 'test_0_pred.conll')

        ts = time.time()
        test_res = list(parser.Predict(options.conll_test1,0))
        te = time.time()
        print 'Finished predicting test_0.', te - ts, 'seconds.'
        utils.write_conll(tespath, test_res)


        tespath = os.path.join(options.output, 'test_1_pred.conll')

        ts = time.time()
        test_res = list(parser.Predict(options.conll_test2,1))
        te = time.time()
        print 'Finished predicting test_1.', te - ts, 'seconds.'
        utils.write_conll(tespath, test_res)

    else:
        print 'Preparing vocab'
        words, w2i, pos1, rels1 = utils.vocab(options.conll_train1)
        _1, _2, pos2, rels2 = utils.vocab(options.conll_train2)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, pos1, pos2, rels1, rels2, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing Hierarchical Tree LSTM parser:'
        parser = easylstm.EasyFirstLSTM(words, pos1, pos2, rels1, rels2, w2i, options)

        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train1,options.conll_train2)
            #devpath = os.path.join(options.output, 'dev_0_epoch_' + str(epoch+1) + '.conll')
            #utils.write_conll(devpath, parser.Predict(options.conll_dev1,0))
            #devpath = os.path.join(options.output, 'dev_1_epoch_' + str(epoch+1) + '.conll')
            #utils.write_conll(devpath, parser.Predict(options.conll_dev2,1))
            parser.Save(os.path.join(options.output, os.path.basename(options.model) + str(epoch+1)))
     #       os.system('perl src/utils/eval.pl -g ' + options.conll_dev  + ' -s ' + devpath  + ' > ' + devpath + '.txt')
