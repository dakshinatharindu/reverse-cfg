import numpy as np

class Node:
    def __init__(self, trace: np.ndarray, input_data=None):
        self.trace = trace            # numpy array
        self.parent = None        # pointer to parent node
        self.left = None            # left child
        self.right = None           # right child
        self.input = input_data       # input data corresponding to the trace
    
    def __repr__(self):
        return f"Node(shape={self.trace.shape}, input={self.input if self.input is not None else 'None'})"
    
class CFG:
    def __init__(self):
        self.root = None
        self.mv_window = 1  # Number of samples to consider for divergence
        self.threshold = 0.05
    
    def display(self):
        def _display_recursive(node: Node, level: int):
            print("  " * level + repr(node))
            for child in [node.left, node.right]:
                if child is not None:
                    _display_recursive(child, level + 1)
        
        if self.root is not None:
            _display_recursive(self.root, 0)

    def _find_divergence_point(self, trace1: np.ndarray, trace2: np.ndarray):
        min_length = min(len(trace1), len(trace2))
        trace1 = trace1[:min_length]
        trace2 = trace2[:min_length]
        
        # Calculate absolute difference
        diff = np.abs(trace1 - trace2)
        
        # Apply moving average if window_size > 1
        if self.mv_window > 1:
            diff = np.convolve(diff, np.ones(self.mv_window)/self.mv_window, mode='same')
        
        # Find points where difference exceeds threshold
        exceeds_threshold = diff > self.threshold
        
        # Look for the point exceeding threshold
        for i in range(len(exceeds_threshold)):
            if exceeds_threshold[i]:
                return i
            
        # No divergence found
        return -1
    
    def insert(self, trace: np.ndarray, input_data):
        new_node = Node(trace, input_data)
        if self.root is None:
            self.root = new_node
        else:
            dv_point = self._find_divergence_point(self.root.trace, new_node.trace)
            self._insert_recursive(self.root, new_node, dv_point)    

    def _insert_recursive(self, current_node: Node, new_node: Node, dv_point=-1):
        if dv_point == -1:
            if current_node.left and current_node.right:
                left_divergence = self._find_divergence_point(current_node.left.trace, new_node.trace[len(current_node.trace):])
                right_divergence = self._find_divergence_point(current_node.right.trace, new_node.trace[len(current_node.trace):])
                if left_divergence > right_divergence:
                    new_node.trace = new_node.trace[len(current_node.trace):]
                    self._insert_recursive(current_node.left, new_node, left_divergence)
                elif right_divergence != -1:
                    new_node.trace = new_node.trace[len(current_node.trace):]
                    self._insert_recursive(current_node.right, new_node, right_divergence)
        else:
            new_parent = Node(current_node.trace[:dv_point], current_node.input)
            current_node.trace = current_node.trace[dv_point:]
            new_node.trace = new_node.trace[dv_point:]

            new_parent.left = current_node
            new_parent.right = new_node
            parent = current_node.parent
            current_node.parent = new_parent
            new_node.parent = new_parent
            if current_node == self.root:
                self.root = new_parent
            else:
                print("Updating parent links")
                if parent.left == current_node:
                    parent.left = new_parent
                else:
                    parent.right = new_parent
                new_parent.parent = parent
            

