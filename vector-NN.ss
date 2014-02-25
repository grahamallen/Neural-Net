#|-------------------------VECTOR NEURAL NET----------------------------|#
#|This multi-layer feed-forward neural net uses vectors to allow for O(1)
  weight updates during backpropagation, and in doing so, we belive it lowers
  the computation time over other list-based neural networks written in scheme.

  To run, this neural net will take use vector called NN (for neural net) that
  will have a specified number of neurons which will be vectors inside of NN.
  These neurons will be equivalent to the layers of nodes in the neural net graph
  figures in our textbook. Each neuron will have a specified number of nodes, which
  will correspond to the nodes of the graph, and each node will be a vector that
  contains a number of input weights dependent on the number of nodes in the prior
  neuron plus a threshold weight for that node.

  By calling NN[0][1][2] in a 0-indexed OO language, we would seek to get the 
  third item of the second node of the first neuron. The order will always be neuron-node-weights

  Really, the NN is just an organized collection of threshold-weight-lists (twls). To
  give a general form of a common example for the multi-layer neural net, consider
  XOR, which could have a corresponding NN vector after learning of:

  #( #( #(5 10 10) #(-15 -10 -10) ) #( #(15 10 10) ) ) ;;XOR threshold weights list

  Note that this looks very similar to the list way of doing a NN. In fact, the only
  difference is the addition of a # in front of every open parenthesis to denote a vector.

  We follow the standard of having each node ordered like so:

  #( threshold-value weight1 weight2 ... weightN ) ;;sample node values

  To run the code, we will provide function names for initialization, forward-propagation,
  back-propagation, learning, and testing in the future.

  Graham and Jules 2/25/2014

  P.S. Here are some helpful gates' twls:

  (define threshold-weights '(((1.5 1 1)))) ;AND
  (define threshold-weights '(((-1.5 -1 -1)))) ;NAND
  (define threshold-weights '(((-.5 -1 -1)))) ;NOR
  (define threshold-weights '(((2 4 4)))) ;OR
  (define threshold-weights '(((5 10 10) (-15 -10 -10)) ((15 10 10)))) ;XOR
  (define threshold-weights '(((1.5 1 1) (-.5 -1 -1)) ((.5 1 1)))) ;NeXOR|#

;(define (randomize Fi) (- (random (/ 4.8 Fi)) (/ (/ 4.8 Fi) 2)))

(define (NN-initialization npnl)                                                        ;To make XOR, pass '(2 2 1) as the nodes-per-neuron-lst (npnl). Note: the first number is for the number of inputs.
  (let* ([inputs (reverse (cdr (reverse npnl)))]                                        ;inputs = '(2 2)
         [weights (map (lambda (x) (+ x 1)) inputs)]                                    ;weights = '(3 3)
         [NN-constructed (vector-map make-vector (list->vector (cdr npnl)) 
                                       (vector-map make-vector (list->vector weights) (make-vector (length inputs) '0)))]
         [randomize (lambda (Fi) (- (random (/ 4.8 Fi)) (/ (/ 4.8 Fi) 2)))]
         [NN-randomized (vector-map (lambda (neurons Fi)
                                      (vector-map (lambda (nodes Fi)
                                                    (vector-map (lambda (weights Fi) (randomize Fi)) nodes (make-vector (vector-length nodes) Fi))) neurons (make-vector (vector-length neurons) Fi))) NN-constructed (list->vector inputs))])
    NN-randomized))

(NN-initialization '(2 2 1))