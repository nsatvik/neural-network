'''
Created on Jan 30, 2013

@author: Satvik
'''
import sys
import math
import random
print('hello,world')

class NeuralNetwork:
    X = []
    y = []   
    Layers = []
    alpha = 0.03 #learning rate
        
    def __init__(self, input_file, output_file, layer_info):
        #Process Input file and populate the X datastructure
        self.process_inputfile(input_file)
        #Process Output Sample file
        self.process_outputfile(output_file)
        
        #init_Layers() # Initializes a 3x3x3x1 Neural Network!
        self.initLayers(layer_info)
        
        
    def initLayers(self, layer_info):
        print('Init Layers')
        for i in range(len(layer_info)):
            self.Layers.append([])
            for j in range(layer_info[i]):
                weights = []
                n = 0
                if (i+1)<(len(layer_info)):
                    n = layer_info[i+1]
                else :
                    n = 0 
                for w in range(n):
                    weights.append(0)
                self.Layers[-1].append([0,0,weights])
        
        #To populate the weights randomly.
        for layer in self.Layers:
            for node in layer:
                n = len(node[2])
                for i in range(len(node[2])):
                    node[2][i] = random.uniform(1,2)
        
    
    def find(self,index): #Gives the output for new input in an array like [1,0,0] 3 digit input
        self.X.append(self.X[index])
        print('Sample ',index,' X is ',self.X[-1],)
        result = self.run(0)
        del self.X[-1]
        return result
        
    def sigmoid(self, x):
        return 1/(1 + math.pow(math.e,-x))
    
    def run(self,isTraining):
        if not isTraining:
            for l in range(len(self.Layers)):
                self.run_forward_propagation(l,-1) #Layer l , last sample!
            res = []
            for node in  self.Layers[-1]:
                res.append(node[0])
            return res
            
        for i in range(len(self.X)):
            for num in range(100):
                for l in range(len(self.Layers)):
                    self.run_forward_propagation(l, i)
                self.run_back_propagation(i)
                self.update_nn_weights()
        
            
    def run_forward_propagation(self, layer, i): # To run the forward prop step with sample i.
        
        if layer == 0:
            for node in self.Layers[layer]:
                for j in range(len(self.X[i])):
                    node[0] = float(self.X[i][j])
        else:
            for i in range(len(self.Layers[layer])):
                self.Layers[layer][i][0] = self.sigmoid(self.compute_wi_Oij(i, layer-1))
    
            
    
    
    def compute_wi_Oij(self, n, layer):
        res = 0.0
        for node in self.Layers[layer]:
            res += node[0]*node[2][n]
        return res
    
    
    def run_back_propagation(self, i): #Run back prop for layer and sample i
        lst = []
        for i in range(len(self.Layers)):
            lst.append(i)
        lst.reverse()
        del lst[-1]
        for l in lst:
            for n in range(len(self.Layers[l])):
                #print('Running back propagation for layer ', l, ' node ', n)
                self.Error_nli(n, l, i)
        
        
    def Error_nli(self, n, layer, i): #layer , node n, sample i
        if layer == (len(self.Layers)-1):
            actual_val = 0
            if n==0:
                actual_val = float(self.y[i])
            else:
                actual_val = 1 - float(self.y[i])
            self.Layers[layer][n][1] =  self.Layers[layer][n][0]*(1-self.Layers[layer][n][0])*(self.Layers[layer][n][0] - actual_val)
            #print("Layer ",layer, ' node ',n, ' error ', self.Layers[layer][n][1])
        else:
            wE = 0
            edges = self.Layers[layer][n][2]
            nextLayer = self.Layers[layer+1]
            for i in range(len(edges)):
                wE += edges[i]*nextLayer[i][1]
            
            self.Layers[layer][n][1] = self.Layers[layer][n][0]*(1-self.Layers[layer][n][0])*wE
            #print("Layer ",layer, ' node ',n, ' output ', self.Layers[layer][n][0],'weight ',wE)
            
    def update_nn_weights(self):
        for l in range(len(self.Layers)-1):
            for n in range(len(self.Layers[l])):
                self.update_weights(l, n)
                
    def update_weights(self, layer, n):
        node = self.Layers[layer][n]
        for i in range(len(node[2])):
            node[2][i] += self.alpha * self.Layers[layer+1][i][1] * node[0]
                            
    def process_inputfile(self, input_file):
        f = open(input_file)
        samples = f.read().split('\n')
        
        for sample in samples:
            self.X.append(sample.split(','))
        
        
        f.close()
    def process_outputfile(self, output_file):
        f = open(output_file)
        self.y = f.read().split('\n')
        
        
    def display(self, type):
        
        if type == 'input':
            print (self.X)
        else:
            print (self.y)
       
    def display_neural_network(self):
       
        for l in range(len(self.Layers)):
            print("Layer ",l," Nodes")
            for node in self.Layers[l]:
                print(node)
        print('-----------------------') 



def main():
    n_network =  NeuralNetwork('input_file.txt','output_file.txt', [3,5,5,1])
    #n_network.display('inpt')
    #n_network.display_neural_network()
    n_network.run(1)
    n_network.display_neural_network()
    #print("Neural Network Trained!")
    i = 5
    for i in range(10):
        print('Expected value : ',n_network.y[i],'Predicted value: ',n_network.find(i))
    #for i in range(1,10):
    #    print('sigmoid(',i,') = ',n_network.sigmoid(i))
    
if __name__ == '__main__' :
    main()