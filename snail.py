#!/usr/bin/python
import NeuralNet
import random

PATH = '/home/*username*/workspace/NeuralNet/Modules/'

# S N A I L
# i e ptn e
# m u pot a
# u r r e r
# l a o l n
# a l a l i
# t   c i n
# e   h g g
# d     e
#       n
#       t


def init():
        random.seed(None)
        size = input('How many neurons in the network? ')
        inputs = 2
        outputs = 1
        ninputs = input('How many inputs per neuron? ')
        noutputs = input('How many outputs per neuron? ')
        binOut = 1
        return NeuralNet.NeuralNet(size, inputs, outputs, ninputs, noutputs, binOut) 

def truthTest(inp, out):
        if out == 1:
                sol = True
        else:
                sol = False
        print inp
        print out
        print sol
        return sol == (inp[0] or inp[1])

def getIns():
        return [random.randint(0, 100) % 2, random.randint(0, 100) % 2]

n = init()

success = 0
for i in xrange(1000000):
        
        ins = getIns()
        n.getinputs(ins)
        
        for i in range(0,n.size):
                n.tick()
        
        if truthTest(ins, n.getoutput(0)):
                n.strengthen(.05)
                success += 1
        else:
                n.weaken(.5, .05, .3, .05)
                success = 0
        
        if success > 99:
                print 'Successful network achieved at size ' + str(n.cost()) 
                break
                
if i > 999998:
        print 'No solution found'
        
sv = raw_input('Would you like to save this network? ')
if sv == "yes":
        name = raw_input('What should the network be called? ')
        print 'Saving network...'
        n.save(PATH + name + '.xml')
