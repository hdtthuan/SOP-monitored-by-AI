class SOPMonitoring:
    """
    Monitors a sequence of actions and ensures they follow the predefined order.
    
    If an action is performed out of order, an error is raised, and the process stops immediately.
    """

    def __init__(self):
        # Define the required SOP sequence
        self.sop_sequence = [
            "get [Clamp Bracket]",
            "get [L-Shaped Bracket]",
            "get screw [1]",
            "get [Allen Key]",
            "Put [Allen Key]",
            "get [Sliding Connector]",
            "get screw [2]",
            "get [Allen Key]",
            "Put [Allen Key]",
            "get [Connector Block]",
            "get screw [3]",
            "get [Allen Key]",
            "Put [Allen Key]",
        ]
        self.current_step = 0  # Track the current position in SOP sequence

    def validate_action(self, action):
        """
        Validates the detected action against the expected SOP sequence.
        
        - If the detected action equals the current expected action, it is valid and the system
          remains in the current step.
        - If the detected action equals the next expected action in the sequence, the system
          transitions to the next step.
        - Otherwise, an error is signaled.
        
        Parameters:
        - action (str): The detected action (e.g., "get [object 1]")
        
        Returns:
        - True if the action is valid (either continuing the current step or properly transitioning).
        - False if the action is out of order.
        """
        # If all steps have been completed
        if self.current_step >= len(self.sop_sequence):
            print("[SOP Completed] All required actions have been performed.")
            return True

        expected_current = self.sop_sequence[self.current_step]

        # If detected action equals the current expected action, it's valid.
        if action == expected_current:
            print(f"[VALID] Continuing expected action: {action}")
            return True

        # If we have a next expected action and the detected action matches it,
        # then transition to the next step.
        if self.current_step + 1 < len(self.sop_sequence):
            expected_next = self.sop_sequence[self.current_step + 1]
            if action == expected_next:
                self.current_step += 1
                print(f"[VALID] Transitioning to next action: {action}")
                return True

        # Otherwise, the detected action is out-of-order.
        next_expected = self.sop_sequence[self.current_step + 1] if self.current_step + 1 < len(self.sop_sequence) else "End"
        print(f"[ERROR] Incorrect action: {action}. Expected: {expected_current} or {next_expected}")
        return False

    def is_sop_complete(self):
        """
        Checks if all SOP actions have been successfully completed.
        
        Returns:
        - True if the current_step has reached the end of sop_sequence.
        - False otherwise.
        """
        return self.current_step >= len(self.sop_sequence)
