(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun sigmoid-prime (x)
  (* x (- 1 x)))

(defun forward-propagate (weights biases inputs)
  (let* ((hidden-layer-inputs (cl-num:dot-product weights :hidden-layer inputs))
         (hidden-layer-outputs (mapcar #'sigmoid hidden-layer-inputs))
         (output-layer-input (cl-num:dot-product weights :output-layer hidden-layer-outputs))
         (output-layer-output (sigmoid output-layer-input)))
    (values output-layer-output hidden-layer-outputs)))

(defun backward-propagate (weights biases inputs target learning-rate)
  (let* ((output, hidden-layer-outputs) (forward-propagate weights biases inputs)
         (output-error (- target output))
         (output-delta (* output-error (sigmoid-prime output)))
         (hidden-layer-deltas (mapcar (lambda (x) (* x (sigmoid-prime x)))
                                       (cl-num:dot-product output-delta weights :output-layer)))
         (weight-deltas :hidden-layer (cl-num:outer-product inputs hidden-layer-deltas)
                        :output-layer (cl-num:outer-product hidden-layer-outputs output-delta))
         (bias-deltas :hidden-layer hidden-layer-deltas
                      :output-layer output-delta))
    (values (cl-num:subtract weights (cl-num:scale learning-rate weight-deltas))
            (cl-num:subtract biases (cl-num:scale learning-rate bias-deltas)))))

(defun train-mlp (weights biases training-data learning-rate epochs)
  (dotimes (epoch epochs)
    (dolist (data training-data)
      (let ((inputs (first data))
            (target (rest data)))
        (multiple-value-bind (new-weights new-biases)
            (backward-propagate weights biases inputs target learning-rate)
          (setf weights new-weights
                biases new-biases)))))
  weights))