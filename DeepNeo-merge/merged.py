# %%
import os, sys, timeit, gzip, glob, numpy, math, cPickle, subprocess, math
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from mlp import relu, HiddenLayer

# %%
def matchDat(afflst, hladic, aadic):
	seqlst = []
	tablst = []
	header = []
	for affin in afflst:
		affstr = affin.strip().split('\t')
		if affstr[0] in hladic:
			hlaseq = hladic[affstr[0]]
			aaseq = affstr[1]
			tmp = []
			tmp0 = []
			for hlain in hlaseq:
				for aain in aaseq:
					if hlain == 'X' or aain=='X':
						tmp0.append([float(0)])
					elif hlain == '*':
						tmp0.append([float(0)])
					elif hlain == '.':
						tmp0.append([float(0)])
					elif aain == 'X':
						tmp0.append([float(0)])
					elif aain == 'U':
						tmp0.append([aadic[hlain, 'C']])
					elif aain == 'J':
						aa1 = aadic[hlain, 'L']
						aa2 = aadic[hlain, 'I']
						aamax = max(aa1, aa2)
						tmp0.append([float(aamax)])
					elif aain == 'Z':
						aa1 = aadic[hlain, 'Q']
						aa2 = aadic[hlain, 'E']
						aamax = max(aa1, aa2)
						tmp0.append([float(aamax)])
					elif aain  == 'B':
						aa1 = aadic[hlain, 'D']
						aa2 = aadic[hlain, 'N']
						aamax = max(aa1, aa2)
						tmp0.append([float(aamax)])
					else:
						tmp0.append([aadic[hlain, aain]])
				tmp.append(tmp0)
				tmp0 = []
			seqlst.append(zip(*tmp))
			tablst.append(int(affstr[2]))
			header.append((affstr[0], affstr[1]))
	seqarray0 = np.array(seqlst, dtype = theano.config.floatX)
	del seqlst
	a_seq2 = seqarray0.reshape(seqarray0.shape[0], seqarray0.shape[1] * seqarray0.shape[2])
	a_lab2 = np.array(tablst, dtype = theano.config.floatX)
	del tablst
	return ((a_seq2, a_lab2)), header
	del a_seq2, a_lab2, header

def HeaderOutput(lstin, outname):
	outw = open(outname, 'w')
	for lin in lstin:
		outw.write('\t'.join(lin)+'\n')
	outw.close()

def modifyMatrix(affydatin_test, seqdatin,outfile):
	hladicin = {x.strip().split('\t')[0]: x.strip().split('\t')[1] for x in open(seqdatin).readlines()}
	aalst = open('data/Calpha.txt').readlines()
	aadicin = {}
	aaseq0 = aalst[0].strip().split('\t')
	for aain in aalst[1:]:
		aastr = aain.strip().split('\t' )
		for i in range(1, len(aastr)):
			aadicin[aaseq0[i-1], aastr[0]] = float(aastr[i])
	afflst = open(affydatin_test).readlines()
	d, test_header = matchDat(afflst, hladicin, aadicin)
	outname0 = affydatin_test
	outname2 = affydatin_test+'.header'
	#np.savez_compressed(outname0, test_seq = test_seq, test_lab = test_lab)
        cPickle.dump(d, gzip.open(outfile, 'wb'), protocol = 2)
	HeaderOutput(test_header, outname2)

Datname = 'data/class1_input.dat'
mhcclass = 'class1'
outputfile = 'temp/class1_input.dat.pkl.gz'

print 'Input file:', Datname
modifyMatrix(Datname, 'data/All_prot_alignseq_C_369.dat',outputfile)
# modifyMatrix(Datname, 'data/MHC2_prot_alignseq.dat',outputfile)
print 'The running is completed!'

# %%
import random
import math
import pylab

def random_mixture_model(pos_mu=.6,pos_sigma=.1,neg_mu=.4,neg_sigma=.1,size=200):
	pos = [(1,random.gauss(pos_mu,pos_sigma),) for x in xrange(size/2)]
	neg = [(0,random.gauss(neg_mu,neg_sigma),) for x in xrange(size/2)]
	return pos+neg

def plot_multiple_rocs_separate(rocList,title='', labels = None, equal_aspect = True):
	""" Plot multiples ROC curves as separate at the same painting area. """
	pylab.clf()
	pylab.title(title)
	for ix, r in enumerate(rocList):
		ax = pylab.subplot(4,4,ix+1)
		pylab.ylim((0,1))
		pylab.xlim((0,1))
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		if equal_aspect:
			cax = pylab.gca()
			cax.set_aspect('equal')
		
		if not labels:
			labels = ['' for x in rocList]
		
		pylab.text(0.2,0.1,labels[ix],fontsize=8)
		pylab.plot([x[0] for x in r.derived_points],[y[1] for y in r.derived_points], 'r-',linewidth=2)
	
	pylab.show()
	
def _remove_duplicate_styles(rocList):
 	""" Checks for duplicate linestyles and replaces duplicates with a random one."""
	pref_styles = ['cx-','mx-','yx-','gx-','bx-','rx-']
	points = 'ov^>+xd'
	colors = 'bgrcmy'
	lines = ['-','-.',':']
	
	rand_ls = []
	
	for r in rocList:
		if r.linestyle not in rand_ls:
			rand_ls.append(r.linestyle)
		else:
			while True:
				if len(pref_styles) > 0:
					pstyle = pref_styles.pop()
					if pstyle not in rand_ls:
						r.linestyle = pstyle
						rand_ls.append(pstyle)
						break
				else:
					ls = ''.join(random.sample(colors,1) + random.sample(points,1)+ random.sample(lines,1))
					if ls not in rand_ls:
						r.linestyle = ls
						rand_ls.append(ls)
						break
						
def plot_multiple_roc(rocList,title='',labels=None, include_baseline=False, equal_aspect=True):
	pylab.clf()
	pylab.ylim((0,1))
	pylab.xlim((0,1))
	pylab.xticks(pylab.arange(0,1.1,.1))
	pylab.yticks(pylab.arange(0,1.1,.1))
	pylab.grid(True)
	if equal_aspect:
		cax = pylab.gca()
		cax.set_aspect('equal')
	pylab.xlabel("1 - Specificity")
	pylab.ylabel("Sensitivity")
	pylab.title(title)
	if not labels:
		labels = [ '' for x in rocList]
	_remove_duplicate_styles(rocList)
	for ix, r in enumerate(rocList):
		pylab.plot([x[0] for x in r.derived_points], [y[1] for y in r.derived_points], r.linestyle, linewidth=1, label=labels[ix])
	if include_baseline:
		pylab.plot([0.0,1.0], [0.0, 1.0], 'k-', label= 'random')
	if labels:
		pylab.legend(loc='lower right')
		
	pylab.show()

def load_decision_function(path):
	fileHandler = open(path,'r')
	reader = fileHandler.readlines()
	reader = [line.strip().split() for line in reader]
	model_data = []
	for line in reader:
		if len(line) == 0: continue
		fClass,fValue = line
		model_data.append((int(fClass), float(fValue)))
	fileHandler.close()

	return model_data
	
class ROCData(object):
	def __init__(self,data,linestyle='rx-'):
		self.data = sorted(data,lambda x,y: cmp(y[1],x[1]))
		self.linestyle = linestyle
		self.auc() #Seed initial points with default full ROC
	
	def auc(self,fpnum=0):
		fps_count = 0
		relevant_pauc = []
		current_index = 0
		max_n = len([x for x in self.data if x[0] == 0])
		if fpnum == 0:
			relevant_pauc = [x for x in self.data]
		elif fpnum > max_n:
			fpnum = max_n
		#Find the upper limit of the data that does not exceed n FPs
		else:
			while fps_count < fpnum:
				relevant_pauc.append(self.data[current_index])
				if self.data[current_index][0] == 0:
					fps_count += 1
				current_index +=1
		total_n = len([x for x in relevant_pauc if x[0] == 0])
		total_p = len(relevant_pauc) - total_n
		
		#Convert to points in a ROC
		previous_df = -1000000.0
		current_index = 0
		points = []
		tp_count, fp_count = 0.0 , 0.0
		tpr, fpr = 0, 0
		while current_index < len(relevant_pauc):
			df = relevant_pauc[current_index][1]
			if previous_df != df:
				points.append((fpr,tpr,fp_count))
			if relevant_pauc[current_index][0] == 0:
				fp_count +=1
			elif relevant_pauc[current_index][0] == 1:
				tp_count +=1
			fpr = fp_count/total_n
			tpr = tp_count/total_p
			previous_df = df
			current_index +=1
		points.append((fpr,tpr,fp_count)) #Add last point
		points.sort(key=lambda i: (i[0],i[1]))
		self.derived_points = points
		
		return self._trapezoidal_rule(points)


	def _trapezoidal_rule(self,curve_pts):
		cum_area = 0.0
		for ix,x in enumerate(curve_pts[0:-1]):
			cur_pt = x
			next_pt = curve_pts[ix+1]
			cum_area += ((cur_pt[1]+next_pt[1])/2.0) * (next_pt[0]-cur_pt[0])
		return cum_area
		
	def calculateStandardError(self,fpnum=0):
		area = self.auc(fpnum)
		
		#real positive cases
		Na =  len([ x for x in self.data if x[0] == 1])
		
		#real negative cases
		Nn =  len([ x for x in self.data if x[0] == 0])
		
		
		Q1 = area / (2.0 - area)
		Q2 = 2 * area * area / (1.0 + area)
		
		return math.sqrt( ( area * (1.0 - area)  +   (Na - 1.0) * (Q1 - area*area) +
						(Nn - 1.0) * (Q2 - area * area)) / (Na * Nn))
							
	
	def plot(self,title='',include_baseline=False,equal_aspect=True):		
		pylab.clf()
		pylab.plot([x[0] for x in self.derived_points], [y[1] for y in self.derived_points], self.linestyle)
		if include_baseline:
			pylab.plot([0.0,1.0], [0.0,1.0],'k-.')
		pylab.ylim((0,1))
		pylab.xlim((0,1))
		pylab.xticks(pylab.arange(0,1.1,.1))
		pylab.yticks(pylab.arange(0,1.1,.1))
		pylab.grid(True)
		if equal_aspect:
			cax = pylab.gca()
			cax.set_aspect('equal')
		pylab.xlabel('1 - Specificity')
		pylab.ylabel('Sensitivity')
		pylab.title(title)
#		pylab.show()
                pylab.savefig(title+'.png')
		
	
	def confusion_matrix(self,threshold,do_print=False):
		pos_points = [x for x in self.data if x[1] >= threshold]
		neg_points = [x for x in self.data if x[1] < threshold]
		tp,fp,fn,tn = self._calculate_counts(pos_points,neg_points)
		if do_print:
			print "\t Actual class"
			print "\t+(1)\t-(0)"
			print "+(1)\t%i\t%i\tPredicted" % (tp,fp)
			print "-(0)\t%i\t%i\tclass" % (fn,tn)
		return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
		

	
	def evaluateMetrics(self,matrix,metric=None,do_print=False):
		accuracy = (matrix['TP'] + matrix['TN'])/ float(sum(matrix.values()))
		sensitivity = (matrix['TP'])/ float(matrix['TP'] + matrix['FN'])
		specificity = (matrix['TN'])/float(matrix['TN'] + matrix['FP'])
		efficiency = (sensitivity + specificity) / 2.0
		positivePredictiveValue =  matrix['TP'] / float(matrix['TP'] + matrix['FP'])
		NegativePredictiveValue = matrix['TN'] / float(matrix['TN'] + matrix['FN'])
		PhiCoefficient = (matrix['TP'] * matrix['TN'] - matrix['FP'] * matrix['FN'])/(
							math.sqrt( (matrix['TP'] + matrix['FP']) *
							           (matrix['TP'] + matrix['FN']) *
									   (matrix['TN'] + matrix['FP']) *
									   (matrix['TN'] + matrix['FN']))) or 1.0
									
		if do_print:
			print 'Sensitivity: ' , sensitivity
			print 'Specificity: ' , specificity
			print 'Efficiency: ' , efficiency
			print 'Accuracy: ' , accuracy
			print 'PositivePredictiveValue: ' , positivePredictiveValue
			print 'NegativePredictiveValue' , NegativePredictiveValue
			print 'PhiCoefficient' , PhiCoefficient
			
		return {'SENS': sensitivity, 'SPEC': specificity, 'ACC': accuracy, 'EFF': efficiency,
				'PPV':positivePredictiveValue, 'NPV':NegativePredictiveValue , 'PHI':  PhiCoefficient}
	
	def _calculate_counts(self,pos_data,neg_data):
		""" Calculates the number of false positives, true positives, false negatives and true negatives """
		tp_count = len([x for x in pos_data if x[0] == 1])
		fp_count = len([x for x in pos_data if x[0] == 0])
		fn_count = len([x for x in neg_data if x[0] == 1])
		tn_count = len([x for x in neg_data if x[0] == 0])
		return tp_count,fp_count,fn_count, tn_count
		
if __name__ == '__main__':
	print "PyRoC - ROC Curve Generator"
	from optparse import OptionParser
	
	parser = OptionParser()
	parser.add_option('-f', '--file', dest='origFile', help="Path to a file with the class and decision function. The first column of each row is the class, and the second the decision score.")
	parser.add_option("-n", "--max fp", dest = "fp_n", default=0, help= "Maximum false positives to calculate up to (for partial AUC).")
	parser.add_option("-p","--plot", action="store_true",dest='plotFlag', default=False, help="Plot the ROC curve (matplotlib required)")
	parser.add_option("-t",'--title', dest= 'ptitle' , default='' , help = 'Title of plot.')
	
	(options,args) = parser.parse_args()

	if (not options.origFile):
		parser.print_help()
		exit()

	df_data = load_decision_function(options.origFile)
	roc = ROCData(df_data)
	roc_n = int(options.fp_n)
	print "ROC AUC: %s" % (str(roc.auc(roc_n)),)
	print 'Standard Error:  %s' % (str(roc.calculateStandardError(roc_n)),) 
	
	print ''
	for pt in roc.derived_points:
		print pt[0],pt[1]
		
	if options.plotFlag:
		roc.plot(options.ptitle,True,True)

# %%
def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def Load_data(dataset):
    print '... loading data'
    train_set, valid_set, test_set = cPickle.load( gzip.open(dataset, 'rb') )
    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y   = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def Load_data_ind(dataset):
    print '... loading data'
    test_set = cPickle.load( gzip.open(dataset, 'rb') )
    test_set_x, test_set_y   = shared_dataset(test_set)
    return [(test_set_x, test_set_y)]

def Load_npdata(dataset):
    print '... loading data'
    datasets = np.load(dataset)
    test_setx = datasets['test_seq']
    test_sety = datasets['test_lab']
    test_set = (test_setx, test_sety)
    test_set_x, test_set_y   = shared_dataset(test_set)
    return [(test_set_x, test_set_y)]

# %%
class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in  = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        W_bound = numpy.sqrt(10. / (fan_in + fan_out))  
        self.W  = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),  borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b   = theano.shared(value=b_values, borrow=True)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),name='W',borrow=True
            )
        if b is None:
            b = theano.shared(
                value=numpy.zeros((n_out,),
                    dtype=theano.config.floatX
                ),name='b',borrow=True
            )
        self.W = W
        self.b = b
        self.output = T.nnet.sigmoid(T.dot(input, self.W)+ self.b).flatten()
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W)+ self.b).flatten()
        self.y_pred = self.output > .5
        self.params = [self.W, self.b]
        self.x = input
    def negative_log_likelihood(self, y):
        return - T.mean(y * T.log(self.output) + (1 - y) * T.log(1. - self.output))

class CNN(object):
    def __init__(self, rng=None, nkerns=None, batch_size=None, in_dim=None, filtsize=None, poolsize=None, hidden=None):
        self.layers = []
        self.params = []
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        layer0_input = self.x.reshape((batch_size, 1, in_dim[0], in_dim[1]))
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, in_dim[0], in_dim[1]),
            filter_shape=(nkerns[0], 1, filtsize[0][0], filtsize[0][1]),
            poolsize=poolsize[0]
        )
        self.layers.append(layer0)
        dim11 = (in_dim[0]-filtsize[0][0] +1)
        dim12 = (in_dim[1]-filtsize[0][1] +1)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], dim11, dim12),
            filter_shape=(nkerns[1], nkerns[0], filtsize[1][0], filtsize[1][1]),
            poolsize=poolsize[1]
        )
        self.layers.append(layer1)
        dim21 = (dim11 - filtsize[1][0] +1)
        dim22 = (dim12 - filtsize[1][1] +1)
        layer2_input = layer1.output.flatten(2)
        layer2 = LogisticRegression(input=layer2_input, n_in=hidden, n_out=1)
        self.layers.append(layer2)
        self.params = [ param for layer in self.layers for param in layer.params ]
        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)
        self.finetune_cost = layer2.negative_log_likelihood(self.y)
    def build_finetune_functions(self, datasets, batch_size, learning_rate, L1_param, L2_param, mom):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y)   = datasets[2]
        index = T.lvector('index')
        gparams = T.grad( self.finetune_cost  + L1_param * abs(self.layers[-1].W).sum() + L2_param * (self.layers[-1].W **2).sum(), self.params)
        updates1 = OrderedDict()
        for param, gparam, gparam_mom in zip(self.params, gparams, self.gparams_mom):
            updates1[gparam_mom] = mom * gparam_mom - learning_rate * gparam
            updates1[param] = param + updates1[gparam_mom]
        train_fn = theano.function(
            inputs =[index],
            outputs=self.finetune_cost,
            updates=updates1,
            givens={ self.x: train_set_x[index],
                     self.y: train_set_y[index]}  )
        valid_pred_fn = theano.function(
            inputs = [index],
            outputs=self.layers[-1].p_y_given_x,
            givens ={self.x: valid_set_x[index]} )
        valid_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: valid_set_y[index] } )
        test_pred_fn = theano.function(
            inputs = [index],
            outputs=self.layers[-1].p_y_given_x,
            givens ={self.x: test_set_x[index]} )
        test_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: test_set_y[index] } )
        def getVals( fn, IDX, n_exp, batch_size ):
            vals = list()
            n_batches = n_exp/ batch_size
            resid     = n_exp  - (n_batches * batch_size)
            cnt = int(math.ceil(batch_size / n_exp))
            for i in range(n_batches):
                vals+= fn(IDX[i*batch_size:(i+1)*batch_size]).tolist()
            if cnt <= 1 and resid !=0:
                val = fn(IDX[(n_batches-1)*batch_size+resid:(n_batches*batch_size)+resid])
                vals+= val[(batch_size-resid):batch_size].tolist()
            if cnt > 1:
                IDX_ = IDX
                for i in range(cnt-1):
                    IDX_ = numpy.concatenate((IDX_, IDX))
                val = fn(IDX_[0: batch_size])
                vals += val[range(n_exp)].tolist()
            return vals
        n_valid_exp = valid_set_x.get_value(borrow=True).shape[0]
        n_test_exp  =  test_set_x.get_value(borrow=True).shape[0]
        def valid_check():
            idx = numpy.random.permutation(range(n_valid_exp))
            valid_y    = getVals( valid_y_fn,   idx, n_valid_exp, batch_size )
            valid_pred = getVals( valid_pred_fn,idx, n_valid_exp, batch_size )
            return valid_y, valid_pred
        def test_check():
            idx = numpy.random.permutation(range(n_test_exp))
            test_y     = getVals( test_y_fn,    idx, n_test_exp, batch_size )
            test_pred  = getVals( test_pred_fn, idx, n_test_exp, batch_size )
            return test_y, test_pred
        return train_fn, valid_check, test_check

# %%
modelFile = 'data/mhc1-pan.pkl.gz'
# modelFile = 'data/mhc2-pan.pkl.gz'
# modelFile = 'data/tcr1-pan.pkl.gz'
# modelFile = 'data/tcr2-pan.pkl.gz'
testdata = 'temp/class1_input.dat.pkl.gz'
predFile = 'temp/class1_mhcbinding_result.txt'

#print '\n', 'Model file: ', modelFile, '\n'
#print 'Test data: ', testdata, '\n'
#print 'Prediction result: ', predFile, '\n'

classifier	= cPickle.load(gzip.open(modelFile))
datasets = Load_data_ind(testdata)
test_set_x, test_set_y = datasets[0]

get_y	= theano.function([], test_set_y)
y_      = get_y()
x_      = np.asarray(test_set_x.get_value(borrow=True) , dtype='float32')

batch_size=int(10)
predict_model = theano.function( inputs = [classifier.x], outputs= classifier.layers[-1].output )
n_exp = ( x_.shape[0] )
cnt = int(math.ceil(batch_size / n_exp))
n_batches = n_exp / batch_size
resid = n_exp - (n_batches * batch_size)
y_answer = list()
y_pred = list()
for index in range(n_batches):
	xx = x_[index * batch_size: (index + 1) * batch_size]
	res = predict_model(xx)
	y_pred += res[range(batch_size)].tolist()

if cnt <= 1 and resid != 0:
	xx = x_[(n_batches-1) * batch_size + resid: (n_batches*batch_size)+resid]
	res = predict_model(xx)
	y_pred += res[(batch_size-resid):batch_size].tolist()

if cnt > 1:
	xx = x_
	for i in range(cnt-1):
		xx = np.concatenate((xx, x_))
	res = predict_model(xx[0: batch_size])
	y_pred += res[range(n_exp)].tolist()

fout = open(predFile,'w')
#tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('/')[-1].split('.')[0]+'.'+testdata.split('/')[-1].split('.')[1]).readlines()]
#tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('.')[0]+'.'+testdata.split('.')[1]).readlines()]
#tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata).readlines()]
tids = ['\t'.join(x.strip().split('\t')[:-1]) for x in open(testdata.split('/')[-1].split('.')[0]+'.'+testdata.split('/')[-1].split('.')[1]).readlines()]
#print(tids)
for i in range(len(y_)):
#        print(tids[i])
	fout.write(tids[i]+'\t'+str(y_pred[i])+'\n')
fout.close()