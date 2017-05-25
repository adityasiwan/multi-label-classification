import os, tempfile, re

class Error(Exception):
    def ___str__(self):
        return self.message

class FieldSeparatorError(Error):
    def __init__(self, fs):
        self.message='Connot find a given and predicted class. Possible FS is not set correctly as %s.' %str(fs)
        self.args=(self.message, fs)

class EmptyError(Error):
    def __init__(self):
        self.message='Confusionmatrix contains no values.'
        self.args=(self.message,)

class ClassError(Error):
    def __init(self, classe):
        self.message = '"%s" is not a known class of the matrix' %str(classe)
        self.args = (self.message, classe)

class TrainingError(Error):
    def __init__(self):
        self.message = 'confusionmatrix does not have information about a training file'
        self.args=(self.message,)

def reformat(instancefile, classfile, confusionmatrixfile=None, classindex=-1, FS=None):
    '''Takes an instancefile and a file with predicted classes and appends the labels to the instance file.
    The outputfile confusionmatrixfile is suited to construct a confusionmatrix.
    Return the path of the confusionmatrixfile.

    Format of instancefile
        feat1 feat2 feat3 class
        feat1 feat2 feat3 class
        ...
    Format of classfile
        class
        class
        ...

    classindex: in the example above classindex is -1 which means the classlabel
    is the last string on every line. When the classlabel is the first on every line (e.g maxent)
    you may specify classindex=0
    FS: field separator in instancefile
    confusionmatrixfile: name of outputfile, if None stores the output in a tempfile
    '''
    t=open(os.path.expanduser(instancefile))
    p=open(os.path.expanduser(classfile))

    if confusionmatrixfile:
        output=os.path.expanduser(confusionmatrixfile)
    else:
        fd, output = tempfile.mkstemp()
        os.close(fd)

    o=open(output, 'w')

    try:
        for l in t:
            parts=l.strip().split(FS)
            pred=p.readline().strip()

            ref = parts.pop(classindex)

            parts.append(ref)
            parts.append(pred)
            o.write(' ' .join(parts)+'\n')
    finally:
        t.close()
        p.close()
        o.close()

    return output

def _read(fname, FS):
    '''Reads in a file fname and returns an iterator. Where
every yielded value is a list of the features in the instances.
The format of the file must be: an instance + gold + predicted on
a newline.
FS: the fieldseparator, if None splits on all whitespace'''
    f=open(os.path.abspath(os.path.expanduser(fname)))

    try:
        for l in f:
            line=l.strip()
            if line and line[0] != '#':
                #Remove distribution
                line = re.sub('{[^{]+?}$', '', line)
                yield line.strip().split(FS)
    finally:
        f.close()

def getConfusionMatrixFromFile(fname, FS=None, training=None):
    '''Returns an instance from ConfusionMatrix for the file fname.
The format of the file must be: an instance + gold + predicted
FS: the fieldseparator, if None splits on all whitespace.
training: the training file (to be able to compute averaged scores like Timbl)'''
    cm = ConfusionMatrix(training=training, FS=FS)
    try:
        for line in _read(fname, FS=FS):
            cm.update(line[-2], line[-1])
    except IndexError:
        raise FieldSeparatorError(FS)
    return cm


def readtraining(fname, classindex=-1, FS=None):
    '''Reads in training and returns a dictionary with the distribution of the
    classes in training'''
    d={}

    f=open(os.path.abspath(os.path.expanduser(fname)))

    try:
        for l in f:
            line=l.strip()
            if line:
                parts = line.split(FS)
                klass = parts[classindex]

                try:
                    d[klass]+=1
                except KeyError:
                    d[klass] = 1
    finally:
        f.close()

    return d



class ConfusionMatrix(object):
    '''A confusionmatrix. Use the update-method to add values.'''
    def __init__(self, training=None, FS=None):
        '''training: the training file
        FS: field separator of training file'''
        self._classes={}
        self._trainingfile = training
        self._training={}

        self._fs = FS

    def __len__(self):
        '''The total number of instances'''
        total=0
        for value in self._classes.values():
            total += sum(value.values())
        return total

    def __repr__(self):
        return '<ConfusionMatrix object based on %d instances>' %len(self)

    def __str__(self):
        if not self.classes:
            raise EmptyError()

        try:
            #All classes (also those that were only predicted but never in the reference!)
            keys = set(self._classes.keys())
            for v in self._classes.values():
                keys.update(set(v.keys()))

            keys = list(keys)
            keys.sort()

            #Add a line of data as should be printed in the confusion matrix
            values=[]
            for key in keys:
                linedata=[]
                d=self._classes.get(key, {})
                for k in keys:
                    linedata.append(d.get(k, 0))
                values.append(tuple([str(key)]+linedata))

            #The format of one line
            #the longest classname
            length = max([len(str(n)) for n in keys]) + 2

            valueformat = ' | %'+str(length) +'d'
            format = '%'+ str(length) + 's'+ valueformat*len(keys)

            output=[]

            #headers
            classformat = ' | %'+str(length) +'s'
            firstlineformat = '  '+ ' '*length + classformat*len(keys)
            output.append(firstlineformat %tuple(keys))
            #a border
            output.append('  '+'-'*len(output[0]))
            predformat = '  %'+str(len(output[0]))+'s'
            output.insert(0, predformat %('predicted'))

            #the ref label
            for i, line in enumerate(values):
                l = format %line
                if i == 0:
                    l = 'r '+l
                elif i == 1:
                    l = 'e '+l
                elif i == 2:
                    l = 'f '+l
                else:
                    l = '  '+l
                output.append(l)

            diff = -1*(len(values) - 3)

            if diff > 0:
                for i in range(diff):
                    output.append('ref'[diff - 1 +i])

            return '\n'.join(output)
        except MemoryError:
            return "Couldn't print confusionmatrix because of limited memory"

    @property
    def classes(self):
        '''the classes in the matrix (testfile)'''
        classes = self._classes.keys()[:]
        classes.sort()
        return classes

    @property
    def training(self):
        '''The classdistribution of training'''
        if not self._trainingfile:
            raise TrainingError()

        if not self._training:
            self._training= readtraining(self._trainingfile, FS=self._fs)
        return self._training

    @property
    def accuracy(self):
        '''Returns the overall accuracy'''
        if not self.classes:
            raise EmptyError()
        return sum([self.TP(c) for c in self.classes]) / float(len(self))

    def update(self, ref, pred):
        '''Updates the confusionmatrix.
ref: the reference class
pred: the predicted class'''
        self._classes[ref][pred] = self._classes.setdefault(ref, {}).setdefault(pred, 0) + 1

    def getattr(self, att):
        return self.__getattribute__(att)

    def reset(self):
        '''Resets the matrix'''
        self._classes={}

    def count(self, classe):
        '''Returns the number of class classe instances in the reference (testfile)'''
        if not self.classes:
            raise EmptyError()
        try:
            return sum(self._classes[classe].values())
        except KeyError:
            raise ClassError(classe)

    def TP(self, classe):
        '''Returns the true positive count for class classe'''
        if not self.classes:
            raise EmptyError()
        try:
            return self._classes[classe][classe]
        except KeyError:
            return 0


    def FP(self, classe):
        '''Returns the false positive count for class classe'''
        if not self.classes:
            raise EmptyError()
        if classe not in self.classes:
            return 0
        fp=0
        for k,v in self._classes.items():
            if k != classe:
                fp+=v.get(classe, 0)
        return fp

    def FN(self, classe):
        '''Returns the false negative count for class classe'''
        if not self.classes:
            raise EmptyError()
        if classe not in self.classes:
            return 0
        fn = 0
        for k,v in self._classes[classe].items():
            if k != classe:
                fn+=v
        return fn

    def TN(self, classe):
        '''Returns the true negative count for class classe'''
        if not self.classes:
            raise EmptyError()
        if classe not in self.classes:
            raise ClassError(classe)
        return len(self) - self.TP(classe) - self.FP(classe) - self.FN(classe)

    def precision(self, classe):
        '''Return the precision of the class classe'''
        tp = float(self.TP(classe))
        fp= self.FP(classe)

        try:
            return tp/(tp+fp)
        except ZeroDivisionError:
            return 0

    def recall(self, classe):
        '''Returns the recall of the class classe'''
        tp = float(self.TP(classe))

        if tp == 0:
            return 0

        fn= self.FN(classe)
        return tp/(tp+fn)

    def fmeasure(self, classe, beta=1.0):
        '''Returns the fmeasure with beta for the class classe:
        (1.0 + beta) * (P*R) / (beta*P + R)'''
        P = self.precision(classe)
        R = self.recall(classe)
        try:
            return (1.0 + beta) * (P*R) / (beta*P + R)
        except ZeroDivisionError:
            return 0

    def macrofmeasure(self, beta=1.0):
        '''The macroaveraged fmeasure
        '''
        if not isinstance(beta, float):
            raise TypeError('beta must be a float or integer')

        try:
            return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.training)
        except TrainingError:
            return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.classes)


    def microfmeasure(self, beta=1.0):
        '''The microaveraged fmeasure.
        '''
        if not isinstance(beta, float):
            raise TypeError('beta must be a float or integer')

        try:
            return sum([self.fmeasure(c, beta=beta)*self.training[c]/sum(self.training.values()) for c in self.training.keys()])
        except TrainingError:
            return sum([self.fmeasure(c, beta=beta)*self.count(c)/len(self) for c in self.classes])






#SHELL SCRIPT++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _main(fname, fs, beta, training=None, printcm=True):
    cm = getConfusionMatrixFromFile(fname, fs, training=training)
    if printcm:
        print cm
        print

    #Calcutalte the length of the longest classlabel
    length= max([len(str(c)) for c in cm.classes])

    print ('%-'+str(length)+'s ') %(' '),'%5s  %5s  %5s  %5s' %('TP', 'FP', 'TN', 'FN')
    for c in cm.classes:
        print ('%-'+str(length)+'s:') %str(c),'%5d  %5d  %5d  %5d' %(cm.TP(c), cm.FP(c), cm.TN(c), cm.FN(c))
    print
    lineformat='precision: %6.4f  recall:  %6.4f   F-measure(beta=%3.1f): %6.4f'
    for c in cm.classes:
        print ('%-'+str(length)+'s:') %str(c), lineformat %(cm.precision(c), cm.recall(c), beta, cm.fmeasure(c, beta))
    print


    output=[]
    if training:
        output.append('Computing averaged scores using train (like Timbl):')
    else:
        output.append('Computing averaged scores using test (may differ slightly from Timbl):')
    #microrecall = sum([cm.recall(cl)*cm.count(cl)/len(cm) for cl in cm.classes])
    #microprec = sum([cm.precision(cl)*cm.count(cl)/len(cm) for cl in cm.classes])
    #output.append( 'Microaveraged recall             : %6.4f' %microrecall)
    #output.append( 'Microaveraged precision          : %6.4f' %microprec)

    output.append( 'Macroaveraged F-measure(beta=%3.1f): %6.4f' %(beta, cm.macrofmeasure(beta)) )
    output.append( 'Microaveraged F-measure(beta=%3.1f): %6.4f' %(beta, cm.microfmeasure(beta)) )
    output.append( 'Overall accuracy                 : %6.4f (%d out of %d)' %(cm.accuracy, sum([cm.TP(c) for c in cm.classes]), len(cm)) )

    print '\n'.join(output)
    #return '\n'.join(output)


def _main2(fname, classfile, classindex, fs, beta, printcm):
    tempfile = reformat(fname, classfile, classindex=classindex, FS=fs)
    try:
        _main(tempfile, fs, beta, printcm=printcm)
    except:
        os.remove(tempfile)
        raise
    else:
        os.remove(tempfile)

def _usage():
    print '''Prints the metrics of a machine learning testfile (version %s)
USAGE:
    ./confusionmatrix.py [-f FS] [-b beta] [-t trainingfile] [-c classindex] file [classfile]

ARGUMENTS:
        file:   a list of tagged instances with at the second last place the reference class and
                at the last place the predicted value (default Timbl)
    OR
        file:       a list of instances with at classindex the reference class
        classfile:  a list of predicted classlabels
        The number of lines in file and classfile must be the same!
        (default maxent)

OPTIONS
    -f FS:   FS is the field separator, defaults to all whitespace. (--field-separator)
    -t trainingfile: For Timbl files you can specify the trainingfile. If given, the script
                     uses this file to compute averaged scores. See NOTE. It is assumed
                     that the last field is the classlabel.
    -b beta: the beta value used to calculate the F-measures (default:1)(--beta)
    -c classindex:  the index of the reference classlabel in file. Starting with 0 (default:the last)
                    Only useful if a classfile is specified (--class)
    -m : Don't print the confusionmatrix. Useful when there are a lot of classes. (--confusionmatrix)

EXAMPLES
    1. TiMBL
    +++++++++
    To use on a Timbl-out file:
        ./confusionmatrix.py -f ',' -t dimin.train dimin.test.IB1.O.gr.k1.out

    2. MAXENT
    ++++++++++
    To use on a maxent test- and outputfile:
        ./confusionmatrix.py -c 0 testfile outputfile

    The outputfile is the file created with the -o option of maxent.

NOTE
    The macro-averaged and micro-averaged f-scores (without teh -t option) may differ from
    the scores that are given by Timbl when using the +v cs option. The -t option is there
    to solve this issue.

    Timbl uses the number of different classes in training for the macro-averaged scores.
    This script uses the number of different classes in test. Therefore, if a class occurs
    in training and not in test the macro-averaged scores of Timbl and this script will differ.
    A difference between Timbl and this script will not be observed very often since most of
    the time all classes of training are also in test and vice versa. Using the -t option will
    resolve this difference.

    Timbl uses the frequency of the classes in training when computing micro-averaged scores whereas
    this script uses the frequencies in test. Because the frequencies in training and test often differ
    the micro-averaged scores of Timbl and this script will also differ often. Using the -t option will
    resolve this difference.


%s, %s
''' %(__version__, __author__, __date__)

if __name__ == '__main__':
    import getopt, sys

    try:
        opts,args=getopt.getopt(sys.argv[1:],'f:hb:c:t:m', ['help', 'confusionmatrix', 'field-separator=', 'beta=', 'class=', 'train='])
    except getopt.GetoptError:
        # print help information and exit:
        _usage()
        sys.exit(2)

    FS=None
    beta=1.0
    classindex=-1
    training=None
    printcm=True

    for o, a in opts:
        if o in ('-h', '--help'):
            _usage()
            sys.exit()
        if o in ('-f', '--field-separator'):
            FS=a
        if o in ('-t', '--train'):
            training=a
        if o in ('-m', '--confusionmatrix'):
            printcm=False
        if o in ('-b', '--beta'):
            try:
                beta=float(a)
            except ValueError:
                print >>sys.stderr, 'Error: beta must be a float.'
                sys.exit(1)
        if o in ('-c', '--class'):
            try:
                classindex=int(a)
            except ValueError:
                print >>sys.stderr, 'Error: classlabel index must be an integer.'
                sys.exit(1)


    #Get control characters for FS
    if FS:
        FS = FS.replace('\\t', '\t')
        FS = FS.replace('\\n', '\n')
        FS = FS.replace('\\r', '\r')


    # 1 or 2 arguments needed
    classfile=None
    if len(args) == 2:
        classfile=args[1]
    elif len(args) > 2:
        _usage()
        sys.exit(2)

    fname = args[0]

    if not os.path.isfile(os.path.expanduser(fname)):
        print >>sys.stderr, 'Error: %s is not an existing file' %fname
        sys.exit(1)

    if training and not os.path.isfile(os.path.expanduser(training)):
        print >>sys.stderr, 'Error: %s is not an existing file' %training
        sys.exit(1)

    if classfile and not os.path.isfile(os.path.expanduser(classfile)):
        print >>sys.stderr, 'Error: %s is not an existing file' %fname
        sys.exit(1)


    try:
        if classfile:
            _main2(fname, classfile, classindex, FS, beta, printcm)
        else:
            _main(fname, FS, beta, training=training, printcm=printcm)
    except Error, e:
        print >>sys.stderr, 'Error:', e.message
        sys.exit(1)
