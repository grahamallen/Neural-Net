#|-------------------------VECTOR NEURAL NET----------------------------
  This multi-layer feed-forward neural net uses vectors to allow for O(1)
  weight updates during backpropagation, and in doing so, we belive it lowers
  the computation time over other list-based neural networks written in scheme.

  To run, this neural net will use a vector called NN (for neural net) that
  will have a specified number of layers which will be vectors inside of NN.
  These layers will be equivalent to the layers of neurons in the neural net graph
  figures in our textbook. (Negnevitsky 2nd Edition) Each layer will have a specified
  number of neurons, which will correspond to the nodes of the graph, and each neuron
  will be a vector that contains a number of input weights dependent on the number of
  neurons in the prior layer and a threshold weight for that neuron.

  By calling NN[0][1][2] in a 0-indexed OO language, we'd seek to get the 
  third item of the second neuron of the first layer. The order will always be layer-neuron-weights.

  Really, the NN is just an organized collection of threshold-weight-vectors (twvs). To
  give a general form of a common example for the multi-layer neural net, consider
  XOR, which could have a corresponding NN vector after learning:

  #( #( #(5 10 10) #(-15 -10 -10) ) #( #(15 10 10) ) ) ;;XOR threshold weights vector

  Note that this looks very similar to the list way of doing a NN. In fact, the only
  difference is the addition of a # in front of every open parenthesis to denote a vector.

  We follow the standard of having each neuron ordered like so:

  #( threshold-value weight1 weight2 ... weightN ) ;;sample neuron values

  To run the code, we provide function names for initialization and activation, and will provide function names for learning and testing in the future.

  Graham and Jules 2/25/2014

  P.S. Here are some helpful gates' twls:

  (define threshold-weights '(((1.5 1 1)))) ;AND
  (define threshold-weights '(((-1.5 -1 -1)))) ;NAND
  (define threshold-weights '(((-.5 -1 -1)))) ;NOR
  (define threshold-weights '(((2 4 4)))) ;OR
  (define threshold-weights '(((5 10 10) (-15 -10 -10)) ((15 10 10)))) ;XOR
  (define threshold-weights '(((1.5 1 1) (-.5 -1 -1)) ((.5 1 1)))) ;NeXOR
---------------------------------------------------------------------------|#

(define npll '(2 2 1)) ;Note: the first number is for the number of initial inputs. To make XOR, pass '(2 2 1) as the neurons-per-layer-lst (npll).
(define NN '#())
(define learning-rate 0.9)

(define (NN-initialize npll)                                                        
  (let* ([num-inputs-lst (reverse (cdr (reverse npll)))]
         [weights (map (lambda (x) (+ x 1)) num-inputs-lst)]
         [NN-constructed (vector-map make-vector (list->vector (cdr npll)) 
                                       (vector-map make-vector (list->vector weights) (make-vector (length num-inputs-lst) '0)))]
         [randomize (lambda (Fi) (- (random (/ 4.8 Fi)) (/ (/ 4.8 Fi) 2)))]
         [NN-randomized (vector-map (lambda (layers Fi)
                                      (vector-map (lambda (neurons Fi)
                                                    (vector-map (lambda (weights Fi) (randomize Fi)) neurons (make-vector (vector-length neurons) Fi))) layers (make-vector (vector-length layers) Fi))) NN-constructed (list->vector num-inputs-lst))])
    (set! NN NN-randomized)))

(define (sigmoid x)
  (/ (+ 1 (exp (- x)))))

(define (perceptron twv inputs)
  (let* ([twl (vector->list twv)]
         [inputs-lst (vector->list inputs)])
    (sigmoid (apply + (map * twl (cons -1 inputs-lst))))))

(define (layer-activate layer inputs)
    (vector-map (lambda (neuron) (perceptron neuron inputs)) layer))

(define (NN-learn inputs desired-outputs)
  (let ([depth 0])
    (NN-feed-forward inputs desired-outputs depth)))

(define (NN-feed-forward inputs desired-outputs depth)
  (if (> (vector-length NN) depth)                                                               ;Do until the last layer in the NN
    (let ([outputs (layer-activate (vector-ref NN depth) inputs)])                               ;Activate each neuron in the layer like a perceptron
      (if (= (- (vector-length NN) 1) depth)                                                     ;If on the last neron, start back-prop
        (NN-back-prop inputs outputs desired-outputs depth)                                      ;set! output layer's new weights and pass new desired-outputs up
        ;else
        (let ([desired-outputs-corrected (NN-feed-forward outputs desired-outputs (+ depth 1))]) ;Recursively call to get corrected desired-outputs
          (NN-back-prop inputs outputs desired-outputs-corrected depth))))))                     ;set! this depth's layer's new weights and pass new desired-outputs up
  
(define (NN-back-prop inputs outputs desired-outputs depth)
  (if (>= depth 0)
    (if (= (- (vector-length NN) 1) depth)
      (let* ([errors (vector-map - desired-outputs outputs)]
             [derivatives (vector-map (lambda (output) (- 1 output)) outputs)]     
             [deltas (vector-map * outputs derivatives errors)]
             [inputs-lst (vector->list inputs)]
             [twv-corrections (vector-map (lambda (delta) 
                                               (vector-map (lambda (i) 
                                                             (* learning-rate i delta)) (list->vector (cons -1 inputs-lst)))) deltas)]
             [twv (vector-ref NN depth)] ;twv = threshold-weight-vector
             [transpose (lambda (matrix) (apply map list matrix))] ;converts a 2D list like '((1 2 3) (4 5 6) (7 8 9)) to '((1 4 7) (2 5 8) (3 6 9))
             [wl (vector->list (vector-map (lambda (x) (cdr (vector->list x))) twv))] ;prunes the threshold values from twv and turns it into a 2D list
             [desired-outputs-corrected (list->vector (map (lambda (n) (apply + (map * n (vector->list deltas)))) (transpose wl)))])
        (vector-set! NN depth (vector-map (lambda (x y) (vector-map + x y)) twv twv-corrections))
        ;(display (format "----UPDATED-OUTPUT-LAYER----\nNN ~a\n" NN))
        desired-outputs-corrected)
      ;else
      (let* ([derivatives (vector-map (lambda (output) (- 1 output)) outputs)]
             [deltas (vector-map * outputs derivatives desired-outputs)] ;Uses the desired-outputs parameter to pass the desired-outputs-corrected values from the previous recursive step.
             [inputs-lst (vector->list inputs)]
             [twv-corrections (vector-map (lambda (delta)
                                            (vector-map (lambda (i)
                                                          (* learning-rate i delta)) (list->vector (cons -1 inputs-lst)))) deltas)]
             [twv (vector-ref NN depth)]
             [transpose (lambda (matrix) (apply map list matrix))] ;converts a 2D list like '((1 2 3) (4 5 6) (7 8 9)) to '((1 4 7) (2 5 8) (3 6 9))
             [wl (vector->list (vector-map (lambda (x) (cdr (vector->list x))) twv))] ;prunes the threshold values from twv and turns it into a 2D list
             [desired-outputs-corrected (list->vector (map (lambda (n) (apply + (map * n (vector->list deltas)))) (transpose wl)))])
        
        (vector-set! NN depth (vector-map (lambda (x y) (vector-map + x y)) twv twv-corrections))
        ;(display (format "----UPDATED-HIDDEN-LAYER----\nNN ~a\n" NN))
        desired-outputs-corrected))))

(define (NN-display epoch)
  (let* ([ins (vector-map (lambda (e) (vector-ref e 0)) epoch)]
         [outs (vector-map (lambda (i) (NN-activate i)) ins)])
    (vector-map (lambda (i o) (display (format "Inputs: ~a Outputs: ~a\n" i o))) ins outs)))

(define (NN-activate inputs)
  (let ([depth 0])
    (NN-activate2 inputs depth)))

(define (NN-activate2 inputs depth)
  (if (> (vector-length NN) depth)
      (let ([outputs (layer-activate (vector-ref NN depth) inputs)])
        (NN-activate2 outputs (+ 1 depth)))
      ;else
      inputs))

(define (NN-iterate epoch total-iterations)
    (letrec ([next-iteration (lambda (epoch i)
      (cond
       [(= i 0)
        (display (format "----RANDOMIZED-NN----\n~a\n" NN))
        (NN-display epoch)
        (next-iteration epoch (+ i 1))]
       [(= i total-iterations)
        (display (format "----LEARNED-NN----\n~a\n" NN))
        (NN-display epoch)
        ]
       [else
        (vector-map (lambda (e) (NN-learn (vector-ref e 0) (vector-ref e 1))) epoch)
        (next-iteration epoch (+ i 1))])
      )])
      (next-iteration epoch 0)))

(NN-initialize npll)
(define epoch '#(#(#(0 0) #(0)) 
                 #(#(0 1) #(1))
                 #(#(1 0) #(1))
                 #(#(1 1) #(0))))
(NN-iterate epoch 1000)