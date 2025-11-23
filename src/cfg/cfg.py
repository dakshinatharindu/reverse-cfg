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
        self.baseline_window = 20
        self.consecutive_points = 3
        self.sigma_multiplier = 4.0
    
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
        diff = np.abs(trace1[:min_length] - trace2[:min_length])

        if self.mv_window > 1:
            diff = np.convolve(diff, np.ones(self.mv_window)/self.mv_window, mode='same')
        
        baseline_values = []  # Rolling window of baseline differences

        consecutive_count = 0
        divergence_idx = -1

        for i in range(len(diff)):
            current_diff = diff[i]
            
            # Only update dynamic threshold if we haven't found divergence yet
            if divergence_idx == -1:
                # Only add to baseline if current diff is less than current dynamic threshold
                if current_diff < self.threshold:
                    baseline_values.append(current_diff)
                    
                    # Keep only the last baseline_window values
                    if len(baseline_values) > self.baseline_window:
                        baseline_values.pop(0)
                    
                    # Update dynamic threshold if we have enough baseline samples
                    if len(baseline_values) >= self.baseline_window:
                        baseline_mean = np.mean(baseline_values)
                        baseline_std = np.std(baseline_values)
                        self.threshold = baseline_mean + self.sigma_multiplier * baseline_std
                        
                        # Ensure threshold is at least the static threshold
                        self.threshold = max(self.threshold, 0.04)

            
            # Check if current point exceeds threshold
            if current_diff > self.threshold:
                consecutive_count += 1
                if consecutive_count >= self.consecutive_points and divergence_idx == -1:
                    # Mark divergence point
                    divergence_idx = i - self.consecutive_points + 1
                    break
            else:
                consecutive_count = 0
        
        return divergence_idx
    
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
                left_divergence = self._find_divergence_point(current_node.left.trace, new_node.trace[len(current_node.left.trace):])
                right_divergence = self._find_divergence_point(current_node.right.trace, new_node.trace[len(current_node.right.trace):])
                if left_divergence > right_divergence:
                    new_node.trace = new_node.trace[len(current_node.left.trace):]
                    self._insert_recursive(current_node.left, new_node, left_divergence)
                elif right_divergence != -1:
                    new_node.trace = new_node.trace[len(current_node.right.trace):]
                    self._insert_recursive(current_node.right, new_node, right_divergence)
        else:
            new_parent = Node(current_node.trace[:dv_point])
            current_node.trace = current_node.trace[dv_point:]
            new_node.trace = new_node.trace[dv_point:]

            new_parent.left = current_node
            new_parent.right = new_node
            current_node.parent = new_parent
            new_node.parent = new_parent
            if current_node == self.root:
                self.root = new_parent
            else:
                parent = current_node.parent
                if parent.left == current_node:
                    parent.left = new_parent
                else:
                    parent.right = new_parent
                new_parent.parent = parent
            

