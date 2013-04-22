#!/usr/bin/python
import random
import math
import xml.etree.ElementTree as xml

class Neuron:
        """Neuron unit in AI"""
               
        def linear(self, num, thresh):
                if num > 1:
                        return 1
                elif num < -1:
                        return -1
                else:
                        return num
        
        def threshold(self, num, thresh):
                if num >= thresh:
                        return 1
                else:
                        return -1
                        
        def piecewise(self, num, thresh):
                if num >= thresh:
                        return 1
                elif num < -thresh:
                        return -1
                else:
                        return num
        
        def sigmoid(self, num, thresh):
                return math.tanh(num)
                 
        def __init__(self, ninputs, activFunc, thresh):
                self.ninputs = ninputs
                self.inputs = [0.0 for count in xrange(ninputs)]
                self.output = 0.0
                self.weights = [random.random()*2.0-1.0 for count in xrange(ninputs)]     
                self.activFunc = activFunc
                self.thresh = thresh
                self.activationFunction = dict({
                        0 : self.linear,
                        1 : self.threshold,
                        2 : self.piecewise,
                        3 : self.sigmoid,
                })
                
        def printNeuron(self):
                print 'Neuron:'
                print 'Activation Function type ' + str(self.activFunc)
                print 'Threshold ' + str(self.thresh)
                for w in self.weights:
                        print w
        
        def tick(self):
                #Reset output
                self.output = 0.0
                
                #Weights and summation
                for i in range(0,self.ninputs):
                        self.output += self.inputs[i]*self.weights[i]
                
                self.output = self.activationFunction[self.activFunc](self.output, self.thresh)
                        
        def resetInputs(self):
                for i in self.inputs:
                        i = 0.0

class NeuralNet:
        """An organized collection of neurons"""
        def __init__(self, size, inputs, outputs, ninputs, noutputs, binOut):
                self.size = size
                self.inputs = inputs
                self.outputs = outputs
                self.ninputs = ninputs
                self.noutputs = noutputs
                self.binOut = binOut
                #net, first size elements are the neural net, next are the outputs, then the inputs
                self.net = [Neuron(ninputs, random.randrange(0, 4), random.random()*2.0 - 1.0) for count in xrange(size + outputs + inputs)]
                if binOut == 1:
                        for neur in range(size, size + outputs):
                                self.net[neur].activFunc = 1
                
                #connections [neuron, input #, strength]
                self.connections = [[random.randrange(0, size + outputs), random.randrange(0, ninputs), random.random()] for count in xrange((size + inputs)*noutputs)]
                
                
        #def __init__(self, location):
                #print location
                
                
        def printNet(self):
                print 'Neural Network'
                for n in self.net:
                        n.printNeuron()
                print 'Connections'
                for c in self.connections:
                        print 'Connected to ' + str(c[0]) + ' at connection # ' + str(c[1]) + ', strength of ' + str(c[2])
        
        def tick(self):
                #print 'Ticking network...'
                neurNum = 0
                
                #first set all of the inputs for the net and outputs
                for n in range(0, self.size + self.outputs):
                        self.net[n].resetInputs()
                
                for c in range(0, len(self.connections)):
                        neurNum = c % (self.size + self.inputs)
                        if neurNum < self.size: #c is in the main network
                                self.net[self.connections[c][0]].inputs[self.connections[c][1]] += self.net[neurNum].output
                        else: #c is an input connection
                                neurNum += self.outputs
                                self.net[self.connections[c][0]].inputs[self.connections[c][1]] += self.net[neurNum].output
                        
                        #make sure that the input to each neuron is between -1, and 1
                        if self.net[self.connections[c][0]].inputs[self.connections[c][1]] > 1:
                                self.net[self.connections[c][0]].inputs[self.connections[c][1]] = 1
                        if self.net[self.connections[c][0]].inputs[self.connections[c][1]] < -1:
                                self.net[self.connections[c][0]].inputs[self.connections[c][1]] = -1
                                
                #then tick each neuron
                for n in self.net:
                        n.tick()
                                
                #finally return outputs
                #print 'Inputs'
                #for i in xrange(len(ins)):
                        #print self.net[self.size + self.outputs + i].inputs[0]
                        
                #print 'Outputs'
                #for o in xrange(self.outputs):
                        #print self.net[self.size+o].output
                
        def getinputs(self, ins):
                for i in xrange(len(ins)):
                        
                        self.net[self.size + self.outputs + i].inputs[0] = ins[i]
                        
        def getoutput(self, out):
                return self.net[self.size + out].output
        
        def strengthen(self, factor):
                print 'Strengthening...'
                for c in self.connections:
                        c[2] += random.random()*factor
                        if c[2] > 1:
                                c[2] = 1
                                
        def weaken(self, factor, change, threshold, randin):
                print 'Weakening...'
                for c in self.connections:
                        c[2] -= random.random()*factor
                        if c[2] < 0: #the connection broke
                                #change the weight
                                self.net[c[0]].weights[c[1]] = random.random() * 2.0 - 1.0
                                
                                #change the activation function
                                if random.random() > change:
                                        self.net[c[0]].activFunc = random.randrange(0, 4)
                                        if self.binOut == 1 and c[0] >= self.size and c[0] < (self.size + self.outputs):
                                                self.net[c[0]].activFunc = 1
                                #change the threshold                
                                if random.random() > threshold:
                                        self.net[c[0]].thresh = random.random() * 2.0 - 1.0
                                        
                                #reconnect to a new neuron
                                c[0] = random.randrange(0, self.size + self.outputs)
                                c[1] = random.randrange(0, self.ninputs)
                                c[2] = random.random()
                                
                #now we see if any inputs should change
                for inp in self.net[self.size + self.outputs:]:
                        if random.random() > randin:
                                inp.weights[0] = random.random() * 2.0 - 1.0
                                
                                if random.random() > change:
                                        inp.activFunc = random.randrange(0, 4)
                                        
                                if random.random() > threshold:
                                        inp.thresh = random.random() * 2.0 - 1.0
                                
        def save(self, location):
                #Root element
                root = xml.Element('NeuralModule')
                network = xml.Element('network')
                root.append(network)
                
                #List the network attributes
                network.attrib['size'] = str(self.size)
                network.attrib['inputs'] = str(self.inputs)
                network.attrib['outputs'] = str(self.outputs)
                network.attrib['ninputs'] = str(self.ninputs)
                network.attrib['noutputs'] = str(self.noutputs)
                network.attrib['binaryout'] = str(self.binOut)
                
                mainNet = xml.Element('net')
                network.append(mainNet)
                               
                #Iterate through all of the neurons in the main network of neurons
                for i in xrange(self.size):
                        neuron = xml.Element('neuron')
                        neuron.attrib['number'] = str(i)
                        mainNet.append(neuron)
                        
                        #weights is the parent of the list of weights
                        weights = xml.Element('weights')
                        neuron.append(weights)
                        
                        #Iterate through all of the weights in the neuron
                        for w in xrange(self.ninputs):
                                weight = xml.Element('weight')
                                weight.text = str(self.net[i].weights[w])
                                weight.attrib['number'] = str(w)
                                weights.append(weight)
                            
                        #append the threshold    
                        threshold = xml.Element('threshold')
                        threshold.text = str(self.net[i].thresh)
                        neuron.append(threshold)
                        
                        #and the activation function
                        actfunction = xml.Element('actfunction')
                        actfunction.text = str(self.net[i].activFunc)
                        neuron.append(actfunction)
                
                
                outNet = xml.Element('output')
                network.append(outNet)   
                #Iterate through all of the output neurons in the network
                for i in xrange(self.size, self.size + self.outputs):
                        neuron = xml.Element('neuron')
                        neuron.attrib['number'] = str(i)
                        outNet.append(neuron)
                        
                        #weights is the parent of the list of weights
                        weights = xml.Element('weights')
                        neuron.append(weights)
                        
                        #Iterate through all of the weights in the neuron
                        for w in xrange(self.ninputs):
                                weight = xml.Element('weight')
                                weight.text = str(self.net[i].weights[w])
                                weight.attrib['number'] = str(w)
                                weights.append(weight)
                            
                        #append the threshold    
                        threshold = xml.Element('threshold')
                        threshold.text = str(self.net[i].thresh)
                        neuron.append(threshold)
                        
                        #and the activation function
                        actfunction = xml.Element('actfunction')
                        actfunction.text = str(self.net[i].activFunc)
                        neuron.append(actfunction)
                 
                        
                inNet = xml.Element('input')
                network.append(inNet)   
                #Iterate through all of the input neurons in the network
                for i in xrange(self.size + self.outputs, self.size + self.outputs + self.inputs):
                        neuron = xml.Element('neuron')
                        neuron.attrib['number'] = str(i)
                        inNet.append(neuron)
                        
                        #weights is the parent of the list of weights
                        weights = xml.Element('weights')
                        neuron.append(weights)
                        
                        #Iterate through all of the weights in the neuron
                        for w in xrange(self.ninputs):
                                weight = xml.Element('weight')
                                weight.text = str(self.net[i].weights[w])
                                weight.attrib['number'] = str(w)
                                weights.append(weight)
                            
                        #append the threshold    
                        threshold = xml.Element('threshold')
                        threshold.text = str(self.net[i].thresh)
                        neuron.append(threshold)
                        
                        #and the activation function
                        actfunction = xml.Element('actfunction')
                        actfunction.text = str(self.net[i].activFunc)
                        neuron.append(actfunction)
                        
                        
                #now it's time for the connections
                xmlConnections = xml.Element('connections')
                root.append(xmlConnections)
                xmlConnections.attrib['number'] = str(len(self.connections))
                #xmlConnect = xml.Element('connection')
                
                for c in xrange(0, len(self.connections)):
                        xmlConnect = xml.Element('connection')
                        #xmlConnections.append(xmlConnect)
                        xmlConnect.attrib['number'] = str(c)
                        xmlConnect.attrib['neuron'] = str(self.connections[c][0])
                        xmlConnect.attrib['input'] = str(self.connections[c][1])
                        xmlConnections.append(xmlConnect)               
               
                #now save it to a file
                file = open(location,'w')
                xml.ElementTree(root).write(file)
                file.close()
                
        def cost(self):
                #rough cost upon sys memory, int = 1, float = 2
                return (3 + 2 * self.ninputs) * (len(self.net)) + 4 * len(self.connections) + 6
                
def load(location):
        print 'Loading neural module from: ' + str(location)
        tree = xml.parse(location)
        root = tree.getroot()
        
        #create neural network based on specs in location
        n = NeuralNet(int(root[0].attrib['size']), int(root[0].attrib['inputs']), int(root[0].attrib['outputs']), int(root[0].attrib['ninputs']), int(root[0].attrib['noutputs']), int(root[0].attrib['binaryout']))
        
        #assign all of the neurons from disk to ram
        for neur in root[0][0]:
                n.net[int(neur.attrib['number'])].thresh = float(root[0][0][int(neur.attrib['number'])][1].text)
                n.net[int(neur.attrib['number'])].activFunc = int(root[0][0][int(neur.attrib['number'])][2].text)
                for weigh in root[0][0][int(neur.attrib['number'])][0]:
                        n.net[int(neur.attrib['number'])].weights[int(weigh.attrib['number'])] = float(weigh.text)
                        
        #now the outputs                
        for neur in root[0][1]:
                n.net[int(neur.attrib['number'])].thresh = float(root[0][1][int(neur.attrib['number']) - n.size][1].text)
                n.net[int(neur.attrib['number'])].activFunc = int(root[0][1][int(neur.attrib['number']) - n.size][2].text)
                for weigh in root[0][1][int(neur.attrib['number']) - n.size][0]:
                        n.net[int(neur.attrib['number'])].weights[int(weigh.attrib['number'])] = float(weigh.text)
                        
        #now the inputs                
        for neur in root[0][2]:
                n.net[int(neur.attrib['number'])].thresh = float(root[0][2][int(neur.attrib['number']) - n.size - n.outputs][1].text)
                n.net[int(neur.attrib['number'])].activFunc = int(root[0][2][int(neur.attrib['number']) - n.size - n.outputs][2].text)
                for weigh in root[0][2][int(neur.attrib['number']) - n.size - n.outputs][0]:
                        n.net[int(neur.attrib['number'])].weights[int(weigh.attrib['number'])] = float(weigh.text)
                                                
                        
        #now for the connections
        for con in root[1]:
                n.connections[int(con.attrib['number'])][0] = int(con.attrib['neuron'])
                n.connections[int(con.attrib['number'])][1] = int(con.attrib['input'])
                n.connections[int(con.attrib['number'])][2] = 1.0
                
        print 'Module loaded successfully'
        return n
        
        
def loadSize(location):
        tree = xml.parse(location)
        root = tree.getroot()
        return int(root[0].attrib['binaryout'])
